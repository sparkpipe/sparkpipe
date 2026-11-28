#include "modules/k3_resident_decode_stage/source/spark_k3_stage_model.cu"
#include "sparkpipe/spark_speculation_tree_frame.h"

#include <algorithm>
#include <math.h>
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

static uint32_t xr_rng = 0x73e1c9d5u;

static uint32_t XrNext(void)
{
	xr_rng = xr_rng * 1664525u + 1013904223u;
	return xr_rng;
}

static uint16_t XrBf16(float value)
{
	uint32_t bits;
	memcpy(&bits, &value, sizeof(bits));
	return (uint16_t)((bits + 0x7fffu + ((bits >> 16u) & 1u)) >> 16u);
}

static uint16_t XrRandBf16(void)
{
	return XrBf16(((float)((int32_t)(XrNext() >> 20u) - 2048)) / 2048.0f);
}

static void XrBuildTreeSpec(uint32_t anchor, uint32_t grain, SparkSpeculationTreeLayout *layout,
	std::vector<uint32_t> &parents, std::vector<uint32_t> &positions, std::vector<uint32_t> &slots,
	std::vector<uint32_t> &forks, std::vector<uint32_t> &offsets, std::vector<uint32_t> &loads, std::vector<uint32_t> &saves)
{
	static const uint32_t parent_init[11] = {
		SPARK_SPECULATION_TREE_FRAME_NO_ROW, 0u, 1u, 2u, 3u, 0u, 5u, 6u, 2u, 8u, 9u
	};
	static const uint32_t token_init[11] = { 10u, 11u, 12u, 13u, 14u, 21u, 22u, 23u, 31u, 32u, 33u };
	const uint32_t rows = 11u;
	XrCheck(SparkSpeculationTreeLayoutBuild(parent_init, token_init, rows, anchor, grain, layout) == SPARK_STATUS_OK, "build multi-branch tree layout");
	parents.assign(parent_init, parent_init + rows);
	positions.resize(rows);
	slots.resize(rows);
	forks.resize(rows);
	offsets.resize(rows);
	loads.assign(rows, 0u);
	saves.assign(rows, 0u);
	for ( uint32_t i = 0u; i < rows; ++i )
	{
		positions[i] = anchor + layout->row[i].depth;
		slots[i] = layout->row[i].slot;
		forks[i] = layout->row[i].fork_position;
		offsets[i] = layout->row[i].slot_offset;
		loads[i] = i != 0u && layout->row[i].branch != layout->row[parent_init[i]].branch ? layout->row[i].branch : 0u;
	}
	for ( uint32_t i = 1u; i < rows; ++i )
		if ( loads[i] != 0u )
			saves[parent_init[i]] |= 1u << (loads[i] - 1u);
}

static void XrTreeKdaVerify(void)
{
	const uint32_t rows = 11u, total_rows = 13u, heads = K3_KDA_HEADS / XR_RANKS;
	const uint32_t qk = heads * K3_KDA_KEY_DIM, vdim = heads * K3_KDA_VALUE_DIM;
	const uint32_t state_bytes = (uint32_t)K3_KDA_RANK_STATE_SLOT_BYTES(heads, 0u);
	const uint64_t win_qk_elems = (uint64_t)qk * K3_KDA_CONV_KERNEL, win_v_elems = (uint64_t)vdim * K3_KDA_CONV_KERNEL;
	SparkSpeculationTreeLayout layout;
	std::vector<uint32_t> parents, positions, slots, forks, offsets, loads, saves;
	XrBuildTreeSpec(120u, XR_RANKS, &layout, parents, positions, slots, forks, offsets, loads, saves);
	std::vector<uint16_t> h_in_q((size_t)total_rows * qk), h_in_k((size_t)total_rows * qk), h_in_v((size_t)total_rows * vdim);
	std::vector<float> h_ret((size_t)total_rows * qk), h_gate((size_t)total_rows * heads);
	std::vector<float> h_wq(win_qk_elems), h_wk(win_qk_elems), h_wv(win_v_elems);
	std::vector<uint16_t> h_win_q0(win_qk_elems), h_win_k0(win_qk_elems), h_win_v0(win_v_elems);
	std::vector<uint8_t> h_state0(state_bytes);
	for ( auto &x : h_in_q ) x = XrRandBf16();
	for ( auto &x : h_in_k ) x = XrRandBf16();
	for ( auto &x : h_in_v ) x = XrRandBf16();
	for ( auto &x : h_ret ) x = 0.70f + 0.25f * ((float)(XrNext() >> 24u) / 256.0f);
	for ( auto &x : h_gate ) x = 0.20f + 0.60f * ((float)(XrNext() >> 24u) / 256.0f);
	for ( auto &x : h_wq ) x = ((float)((int32_t)(XrNext() >> 20u) - 2048)) / 4096.0f;
	for ( auto &x : h_wk ) x = ((float)((int32_t)(XrNext() >> 20u) - 2048)) / 4096.0f;
	for ( auto &x : h_wv ) x = ((float)((int32_t)(XrNext() >> 20u) - 2048)) / 4096.0f;
	for ( auto &x : h_win_q0 ) x = XrRandBf16();
	for ( auto &x : h_win_k0 ) x = XrRandBf16();
	for ( auto &x : h_win_v0 ) x = XrRandBf16();
	for ( uint32_t i = 0u; i < state_bytes / sizeof(float); ++i )
	{
		float val = ((float)((int32_t)(XrNext() >> 20u) - 2048)) / 4096.0f;
		memcpy(h_state0.data() + i * sizeof(float), &val, sizeof(float));
	}
	uint8_t *d_state = (uint8_t *)XrDevice(state_bytes);
	uint16_t *d_win_q = (uint16_t *)XrDevice(win_qk_elems * 2u), *d_win_k = (uint16_t *)XrDevice(win_qk_elems * 2u), *d_win_v = (uint16_t *)XrDevice(win_v_elems * 2u);
	float *d_wq = (float *)XrDevice(win_qk_elems * 4u), *d_wk = (float *)XrDevice(win_qk_elems * 4u), *d_wv = (float *)XrDevice(win_v_elems * 4u);
	uint16_t *d_rq = (uint16_t *)XrDevice((size_t)total_rows * qk * 2u), *d_rk = (uint16_t *)XrDevice((size_t)total_rows * qk * 2u), *d_rv = (uint16_t *)XrDevice((size_t)total_rows * vdim * 2u);
	uint16_t *d_q = (uint16_t *)XrDevice((size_t)total_rows * qk * 2u), *d_k = (uint16_t *)XrDevice((size_t)total_rows * qk * 2u), *d_v = (uint16_t *)XrDevice((size_t)total_rows * vdim * 2u);
	uint16_t *d_out = (uint16_t *)XrDevice((size_t)total_rows * vdim * 2u);
	float *d_ret = (float *)XrDevice((size_t)total_rows * qk * 4u), *d_gate = (float *)XrDevice((size_t)total_rows * heads * 4u);
	float *d_snap = (float *)XrDevice(LmDeltaRuleSnapshotBytes(heads, K3_KDA_KEY_DIM, K3_KDA_VALUE_DIM));
	uint32_t *d_idx = (uint32_t *)XrDevice(sizeof(uint32_t)), *d_begin = (uint32_t *)XrDevice(2u * sizeof(uint32_t));
	uint32_t *d_order = (uint32_t *)XrDevice(rows * sizeof(uint32_t)), *d_parent = (uint32_t *)XrDevice(rows * sizeof(uint32_t));
	uint32_t *d_loads = (uint32_t *)XrDevice(rows * sizeof(uint32_t)), *d_saves = (uint32_t *)XrDevice(rows * sizeof(uint32_t));
	uint32_t *d_kept = (uint32_t *)XrDevice(sizeof(uint32_t));
	const uint32_t h_idx = 0u, h_begin[2] = { 0u, rows };
	std::vector<uint32_t> h_order(rows);
	for ( uint32_t i = 0u; i < rows; ++i ) h_order[i] = i;
	cudaMemcpy(d_wq, h_wq.data(), win_qk_elems * 4u, cudaMemcpyHostToDevice);
	cudaMemcpy(d_wk, h_wk.data(), win_qk_elems * 4u, cudaMemcpyHostToDevice);
	cudaMemcpy(d_wv, h_wv.data(), win_v_elems * 4u, cudaMemcpyHostToDevice);
	cudaMemcpy(d_rq, h_in_q.data(), (size_t)total_rows * qk * 2u, cudaMemcpyHostToDevice);
	cudaMemcpy(d_rk, h_in_k.data(), (size_t)total_rows * qk * 2u, cudaMemcpyHostToDevice);
	cudaMemcpy(d_rv, h_in_v.data(), (size_t)total_rows * vdim * 2u, cudaMemcpyHostToDevice);
	cudaMemcpy(d_ret, h_ret.data(), (size_t)total_rows * qk * 4u, cudaMemcpyHostToDevice);
	cudaMemcpy(d_gate, h_gate.data(), (size_t)total_rows * heads * 4u, cudaMemcpyHostToDevice);
	cudaMemcpy(d_idx, &h_idx, sizeof(uint32_t), cudaMemcpyHostToDevice);
	cudaMemcpy(d_begin, h_begin, 2u * sizeof(uint32_t), cudaMemcpyHostToDevice);
	cudaMemcpy(d_order, h_order.data(), rows * sizeof(uint32_t), cudaMemcpyHostToDevice);
	cudaMemcpy(d_parent, parents.data(), rows * sizeof(uint32_t), cudaMemcpyHostToDevice);
	cudaMemcpy(d_loads, loads.data(), rows * sizeof(uint32_t), cudaMemcpyHostToDevice);
	cudaMemcpy(d_saves, saves.data(), rows * sizeof(uint32_t), cudaMemcpyHostToDevice);
	auto reset_kda = [&]()
	{
		cudaMemcpy(d_state, h_state0.data(), state_bytes, cudaMemcpyHostToDevice);
		cudaMemcpy(d_win_q, h_win_q0.data(), win_qk_elems * 2u, cudaMemcpyHostToDevice);
		cudaMemcpy(d_win_k, h_win_k0.data(), win_qk_elems * 2u, cudaMemcpyHostToDevice);
		cudaMemcpy(d_win_v, h_win_v0.data(), win_v_elems * 2u, cudaMemcpyHostToDevice);
	};
	reset_kda();
	const LmCausalConvStreams<float> tree_conv = {
		{ d_win_q, d_win_k, d_win_v }, { d_rq, d_rk, d_rv }, { d_wq, d_wk, d_wv }, { d_q, d_k, d_v }, { qk, qk, vdim }
	};
	XrCheck(LmCausalConvStreamsLaunch<K3_KDA_CONV_KERNEL, LM_CONV_SWISH, float>(tree_conv, d_idx, d_begin, 0, 1u, 0u, d_order, rows, 0, d_parent) == cudaSuccess,
		"KDA tree conv streams launch");
	LM_LAUNCH((LmL2NormalisePerHeadKernel<K3_LAYER_THREADS, K3_KDA_KEY_DIM>), dim3(rows, heads), K3_LAYER_THREADS, 0, 0, d_q, heads, rows, K3_RMS_EPSILON);
	LM_LAUNCH((LmL2NormalisePerHeadKernel<K3_LAYER_THREADS, K3_KDA_KEY_DIM>), dim3(rows, heads), K3_LAYER_THREADS, 0, 0, d_k, heads, rows, K3_RMS_EPSILON);
	XrCheck(LmDeltaRuleLaunch<K3_LAYER_THREADS, K3_KDA_KEY_DIM, K3_KDA_VALUE_DIM, float>(rows, d_state, state_bytes, d_idx, d_begin, 0,
		d_q, d_k, d_v, d_ret, d_gate, d_out, heads, 1u, 1u, 0u, d_order, 0, d_loads, d_saves, d_snap) == cudaSuccess, "KDA tree delta launch");
	XrCheck(cudaDeviceSynchronize() == cudaSuccess, "KDA tree sync");
	std::vector<uint16_t> tree_out((size_t)rows * vdim), check_wq(win_qk_elems), check_wk(win_qk_elems), check_wv(win_v_elems);
	std::vector<uint8_t> check_state(state_bytes);
	cudaMemcpy(tree_out.data(), d_out, tree_out.size() * 2u, cudaMemcpyDeviceToHost);
	cudaMemcpy(check_state.data(), d_state, state_bytes, cudaMemcpyDeviceToHost);
	cudaMemcpy(check_wq.data(), d_win_q, win_qk_elems * 2u, cudaMemcpyDeviceToHost);
	cudaMemcpy(check_wk.data(), d_win_k, win_qk_elems * 2u, cudaMemcpyDeviceToHost);
	cudaMemcpy(check_wv.data(), d_win_v, win_v_elems * 2u, cudaMemcpyDeviceToHost);
	XrCheck(memcmp(check_state.data(), h_state0.data(), state_bytes) == 0 &&
		memcmp(check_wq.data(), h_win_q0.data(), win_qk_elems * 2u) == 0 &&
		memcmp(check_wk.data(), h_win_k0.data(), win_qk_elems * 2u) == 0 &&
		memcmp(check_wv.data(), h_win_v0.data(), win_v_elems * 2u) == 0,
		"KDA tree verify with commit=0 leaves state and windows bit-for-bit untouched");
	auto run_seq_step = [&](uint32_t row_id, uint16_t *out_host)
	{
		const uint32_t one_begin[2] = { 0u, 1u };
		cudaMemcpy(d_begin, one_begin, 2u * sizeof(uint32_t), cudaMemcpyHostToDevice);
		cudaMemcpy(d_order, &row_id, sizeof(uint32_t), cudaMemcpyHostToDevice);
		const LmCausalConvStreams<float> seq_conv = {
			{ d_win_q, d_win_k, d_win_v },
			{ d_rq + (size_t)row_id * qk, d_rk + (size_t)row_id * qk, d_rv + (size_t)row_id * vdim },
			{ d_wq, d_wk, d_wv },
			{ d_q, d_k, d_v },
			{ qk, qk, vdim }
		};
		XrCheck(LmCausalConvStreamsLaunch<K3_KDA_CONV_KERNEL, LM_CONV_SWISH, float>(seq_conv, d_idx, d_begin, 0, 1u, 1u, 0, 1u, 0, 0) == cudaSuccess, "KDA seq conv");
		LM_LAUNCH((LmL2NormalisePerHeadKernel<K3_LAYER_THREADS, K3_KDA_KEY_DIM>), dim3(1u, heads), K3_LAYER_THREADS, 0, 0, d_q, heads, 1u, K3_RMS_EPSILON);
		LM_LAUNCH((LmL2NormalisePerHeadKernel<K3_LAYER_THREADS, K3_KDA_KEY_DIM>), dim3(1u, heads), K3_LAYER_THREADS, 0, 0, d_k, heads, 1u, K3_RMS_EPSILON);
		XrCheck(LmDeltaRuleLaunch<K3_LAYER_THREADS, K3_KDA_KEY_DIM, K3_KDA_VALUE_DIM, float>(1u, d_state, state_bytes, d_idx, d_begin, 0,
			d_q, d_k, d_v, d_ret + (size_t)row_id * qk, d_gate + (size_t)row_id * heads, d_out, heads, 1u, 1u, 1u, 0, 0) == cudaSuccess, "KDA seq delta");
		XrCheck(cudaDeviceSynchronize() == cudaSuccess, "KDA seq sync");
		cudaMemcpy(out_host, d_out, vdim * 2u, cudaMemcpyDeviceToHost);
	};
	const std::vector<std::vector<uint32_t>> paths = {
		{ 0u, 1u, 2u, 3u, 4u },
		{ 0u, 5u, 6u, 7u },
		{ 0u, 1u, 2u, 8u, 9u, 10u }
	};
	uint32_t kda_tree_diffs = 0u;
	for ( uint32_t p = 0u; p < paths.size(); ++p )
	{
		reset_kda();
		std::vector<uint16_t> seq_out(vdim);
		for ( uint32_t row_id : paths[p] )
		{
			run_seq_step(row_id, seq_out.data());
			if ( memcmp(tree_out.data() + (size_t)row_id * vdim, seq_out.data(), vdim * 2u) != 0 )
				kda_tree_diffs++;
		}
	}
	XrCheck(kda_tree_diffs == 0u, "KDA tree verify matches sequential decode bit-for-bit across trunk and both off-trunk branches");
	const std::vector<std::vector<uint32_t>> accept_paths = {
		{ 0u, 1u, 2u, 3u },
		{ 0u, 1u, 2u, 8u, 9u, 10u }
	};
	for ( uint32_t a = 0u; a < accept_paths.size(); ++a )
	{
		const auto &ap = accept_paths[a];
		const uint32_t kept = (uint32_t)ap.size();
		const uint32_t path_begin[2] = { 0u, kept };
		reset_kda();
		cudaMemcpy(d_begin, path_begin, 2u * sizeof(uint32_t), cudaMemcpyHostToDevice);
		cudaMemcpy(d_order, ap.data(), kept * sizeof(uint32_t), cudaMemcpyHostToDevice);
		cudaMemcpy(d_kept, &kept, sizeof(uint32_t), cudaMemcpyHostToDevice);
		LmCausalConvLaunch<K3_KDA_CONV_KERNEL, LM_CONV_SWISH, float>(d_win_q, d_idx, d_begin, d_kept, d_rq, d_wq, d_q, qk, 1u, 1u, d_order, rows, 0);
		LmCausalConvLaunch<K3_KDA_CONV_KERNEL, LM_CONV_SWISH, float>(d_win_k, d_idx, d_begin, d_kept, d_rk, d_wk, d_k, qk, 1u, 1u, d_order, rows, 0);
		LmCausalConvLaunch<K3_KDA_CONV_KERNEL, LM_CONV_SWISH, float>(d_win_v, d_idx, d_begin, d_kept, d_rv, d_wv, d_v, vdim, 1u, 1u, d_order, rows, 0);
		LM_LAUNCH((LmL2NormalisePerHeadKernel<K3_LAYER_THREADS, K3_KDA_KEY_DIM>), dim3(rows, heads), K3_LAYER_THREADS, 0, 0, d_q, heads, rows, K3_RMS_EPSILON);
		LM_LAUNCH((LmL2NormalisePerHeadKernel<K3_LAYER_THREADS, K3_KDA_KEY_DIM>), dim3(rows, heads), K3_LAYER_THREADS, 0, 0, d_k, heads, rows, K3_RMS_EPSILON);
		LM_LAUNCH((LmDeltaRuleColumnKernel<K3_LAYER_THREADS, K3_KDA_KEY_DIM, K3_KDA_VALUE_DIM, float>), dim3(1u, heads, K3_KDA_VALUE_DIM / LM_WARP_LANES),
			LM_DELTA_COLUMN_THREADS, 0, 0, d_state, state_bytes, d_idx, d_begin, d_kept, d_q, d_k, d_v, d_ret, d_gate, d_out, heads, 1u, 1u, 1u, d_order, 0, 0, 0);
		XrCheck(cudaDeviceSynchronize() == cudaSuccess, "KDA KeepPath fold sync");
		std::vector<uint16_t> keep_next1(vdim), keep_next2(vdim), seq_next1(vdim), seq_next2(vdim), seq_tmp(vdim);
		std::vector<uint8_t> keep_state(state_bytes), seq_state(state_bytes);
		std::vector<uint16_t> keep_wq(win_qk_elems), seq_wq(win_qk_elems), keep_wv(win_v_elems), seq_wv(win_v_elems);
		run_seq_step(11u, keep_next1.data());
		run_seq_step(12u, keep_next2.data());
		cudaMemcpy(keep_state.data(), d_state, state_bytes, cudaMemcpyDeviceToHost);
		cudaMemcpy(keep_wq.data(), d_win_q, win_qk_elems * 2u, cudaMemcpyDeviceToHost);
		cudaMemcpy(keep_wv.data(), d_win_v, win_v_elems * 2u, cudaMemcpyDeviceToHost);
		reset_kda();
		for ( uint32_t row_id : ap )
			run_seq_step(row_id, seq_tmp.data());
		run_seq_step(11u, seq_next1.data());
		run_seq_step(12u, seq_next2.data());
		cudaMemcpy(seq_state.data(), d_state, state_bytes, cudaMemcpyDeviceToHost);
		cudaMemcpy(seq_wq.data(), d_win_q, win_qk_elems * 2u, cudaMemcpyDeviceToHost);
		cudaMemcpy(seq_wv.data(), d_win_v, win_v_elems * 2u, cudaMemcpyDeviceToHost);
		XrCheck(memcmp(keep_next1.data(), seq_next1.data(), vdim * 2u) == 0 &&
			memcmp(keep_next2.data(), seq_next2.data(), vdim * 2u) == 0 &&
			memcmp(keep_state.data(), seq_state.data(), state_bytes) == 0 &&
			memcmp(keep_wq.data(), seq_wq.data(), win_qk_elems * 2u) == 0 &&
			memcmp(keep_wv.data(), seq_wv.data(), win_v_elems * 2u) == 0,
			a == 0u ? "KDA KeepPath on trunk + 2 continuation steps match sequential decode bit-for-bit"
				: "KDA KeepPath on off-trunk branch + 2 continuation steps match sequential decode bit-for-bit");
	}
	printf("TREE-KDA: 11 rows (trunk + 2 branches), KeepPath (trunk & off-trunk) + 2 post-KeepPath steps: 0 bit diffs\n");
	cudaFree(d_state); cudaFree(d_win_q); cudaFree(d_win_k); cudaFree(d_win_v);
	cudaFree(d_wq); cudaFree(d_wk); cudaFree(d_wv);
	cudaFree(d_rq); cudaFree(d_rk); cudaFree(d_rv);
	cudaFree(d_q); cudaFree(d_k); cudaFree(d_v); cudaFree(d_out);
	cudaFree(d_ret); cudaFree(d_gate); cudaFree(d_snap);
	cudaFree(d_idx); cudaFree(d_begin); cudaFree(d_order);
	cudaFree(d_parent); cudaFree(d_loads); cudaFree(d_saves); cudaFree(d_kept);
}

static void XrTreeMlaDsaVerify(uint32_t anchor, uint32_t indexed)
{
	const uint32_t rows = 11u, total_rows = 13u, heads_per_rank = K3_MLA_HEADS / XR_RANKS;
	const uint32_t slot_elems = K3GlobalKv::kSlotBytes / 2u;
	SparkSpeculationTreeLayout layout;
	std::vector<uint32_t> parents, positions, slots, forks, offsets, loads, saves;
	XrBuildTreeSpec(anchor, XR_RANKS, &layout, parents, positions, slots, forks, offsets, loads, saves);
	const uint32_t max_pos = layout.context + 8u;
	const uint32_t pages = (uint32_t)K3GlobalKv::PagesForTokens(max_pos);
	const uint64_t mla_page_bytes = LmKvShardPageStride<K3GlobalKv>(SparkK3KvShardContext(0u, XR_RANKS));
	const uint64_t idx_page_bytes = LmKvShardPageStride<K3IndexKv>(SparkK3KvShardContext(0u, XR_RANKS));
	const uint64_t query_stride = SparkK3KvShardQueryStride(rows, XR_RANKS);
	const uint64_t partial_stride = SparkK3KvShardPartialStride(rows, XR_RANKS);
	const uint32_t index_stride = SparkKvShardGatherKeys(SparkK3KvShardContext(0u, XR_RANKS), max_pos);
	const uint32_t index_keep = index_stride < K3_INDEX_TOPK ? index_stride : K3_INDEX_TOPK;
	const uint32_t bitmap_words = LmIndexSelectBitmapWords(max_pos);
	const float index_scale = K3_INDEX_SCALE / sqrtf((float)K3_INDEX_HEADS);

	std::vector<uint32_t> h_table(pages);
	for ( uint32_t p = 0u; p < pages; ++p ) h_table[p] = (p * 5u + 1u) % pages;
	std::vector<uint16_t> h_prefix_mla((size_t)anchor * slot_elems), h_prefix_idx((size_t)anchor * K3_INDEX_DIM);
	std::vector<uint16_t> h_tree_mla((size_t)total_rows * slot_elems), h_tree_idx((size_t)total_rows * K3_INDEX_DIM);
	std::vector<uint16_t> h_query_all((size_t)XR_RANKS * query_stride, 0u);
	std::vector<uint16_t> h_idx_q((size_t)total_rows * K3_INDEX_QUERY), h_idx_w((size_t)total_rows * K3_INDEX_HEADS);
	for ( auto &x : h_prefix_mla ) x = XrRandBf16();
	for ( auto &x : h_prefix_idx ) x = XrRandBf16();
	for ( auto &x : h_tree_mla ) x = XrRandBf16();
	for ( auto &x : h_tree_idx ) x = XrRandBf16();
	for ( auto &x : h_idx_q ) x = XrRandBf16();
	for ( auto &x : h_idx_w ) x = XrBf16(0.25f + 0.5f * ((float)(XrNext() >> 24u) / 256.0f));
	for ( uint32_t r = 0u; r < XR_RANKS; ++r )
		for ( uint32_t row = 0u; row < total_rows; ++row )
			for ( uint32_t e = 0u; e < heads_per_rank * K3_MLA_KV_A_DIM; ++e )
				if ( row < rows )
					h_query_all[(size_t)r * query_stride + (size_t)row * heads_per_rank * K3_MLA_KV_A_DIM + e] = XrRandBf16();
	std::vector<uint16_t> h_cont_query((size_t)2u * XR_RANKS * heads_per_rank * K3_MLA_KV_A_DIM);
	for ( auto &x : h_cont_query ) x = XrRandBf16();

	uint32_t *d_table = (uint32_t *)XrDevice((uint64_t)pages * sizeof(uint32_t));
	cudaMemcpy(d_table, h_table.data(), (size_t)pages * sizeof(uint32_t), cudaMemcpyHostToDevice);
	uint16_t *d_prefix_mla = (uint16_t *)XrDevice(h_prefix_mla.size() * 2u);
	uint16_t *d_prefix_idx = (uint16_t *)XrDevice(h_prefix_idx.size() * 2u);
	uint16_t *d_tree_mla = (uint16_t *)XrDevice(h_tree_mla.size() * 2u);
	uint16_t *d_tree_idx = (uint16_t *)XrDevice(h_tree_idx.size() * 2u);
	uint16_t *d_query_all = (uint16_t *)XrDevice(h_query_all.size() * 2u);
	uint16_t *d_idx_q = (uint16_t *)XrDevice(h_idx_q.size() * 2u);
	uint16_t *d_idx_w = (uint16_t *)XrDevice(h_idx_w.size() * 2u);
	cudaMemcpy(d_prefix_mla, h_prefix_mla.data(), h_prefix_mla.size() * 2u, cudaMemcpyHostToDevice);
	cudaMemcpy(d_prefix_idx, h_prefix_idx.data(), h_prefix_idx.size() * 2u, cudaMemcpyHostToDevice);
	cudaMemcpy(d_tree_mla, h_tree_mla.data(), h_tree_mla.size() * 2u, cudaMemcpyHostToDevice);
	cudaMemcpy(d_tree_idx, h_tree_idx.data(), h_tree_idx.size() * 2u, cudaMemcpyHostToDevice);
	cudaMemcpy(d_query_all, h_query_all.data(), h_query_all.size() * 2u, cudaMemcpyHostToDevice);
	cudaMemcpy(d_idx_q, h_idx_q.data(), h_idx_q.size() * 2u, cudaMemcpyHostToDevice);
	cudaMemcpy(d_idx_w, h_idx_w.data(), h_idx_w.size() * 2u, cudaMemcpyHostToDevice);

	std::vector<uint32_t> h_prefix_seq(anchor, 0u), h_prefix_pos(anchor);
	for ( uint32_t i = 0u; i < anchor; ++i ) h_prefix_pos[i] = i;
	uint32_t *d_prefix_seq = (uint32_t *)XrDevice((uint64_t)anchor * 4u);
	uint32_t *d_prefix_pos = (uint32_t *)XrDevice((uint64_t)anchor * 4u);
	cudaMemcpy(d_prefix_seq, h_prefix_seq.data(), (size_t)anchor * 4u, cudaMemcpyHostToDevice);
	cudaMemcpy(d_prefix_pos, h_prefix_pos.data(), (size_t)anchor * 4u, cudaMemcpyHostToDevice);

	std::vector<uint32_t> h_seq(rows, 0u);
	uint32_t *d_seq = (uint32_t *)XrDevice((uint64_t)rows * 4u);
	uint32_t *d_pos = (uint32_t *)XrDevice((uint64_t)rows * 4u);
	uint32_t *d_slots = (uint32_t *)XrDevice((uint64_t)rows * 4u);
	uint32_t *d_forks = (uint32_t *)XrDevice((uint64_t)rows * 4u);
	uint32_t *d_offsets = (uint32_t *)XrDevice((uint64_t)rows * 4u);
	uint32_t *d_ctx = (uint32_t *)XrDevice(4u);
	uint32_t *d_path = (uint32_t *)XrDevice((uint64_t)rows * 4u);
	uint32_t *d_kept = (uint32_t *)XrDevice(4u);
	cudaMemcpy(d_seq, h_seq.data(), (size_t)rows * 4u, cudaMemcpyHostToDevice);
	cudaMemcpy(d_pos, positions.data(), (size_t)rows * 4u, cudaMemcpyHostToDevice);
	cudaMemcpy(d_slots, slots.data(), (size_t)rows * 4u, cudaMemcpyHostToDevice);
	cudaMemcpy(d_forks, forks.data(), (size_t)rows * 4u, cudaMemcpyHostToDevice);
	cudaMemcpy(d_offsets, offsets.data(), (size_t)rows * 4u, cudaMemcpyHostToDevice);

	struct RankBuf
	{
		uint8_t *mla_pool;
		uint8_t *idx_pool;
		LmKvAccessError *mla_err;
		LmKvAccessError *idx_err;
		LmFrameError *sel_err;
		LmKvShardView mla_view;
		LmKvShardView idx_view;
		float *partials_send;
		float *partials_recv;
		uint16_t *out_bf16;
		float *scores;
		uint2 *idx_send;
	};
	std::vector<RankBuf> ranks(XR_RANKS);
	for ( uint32_t r = 0u; r < XR_RANKS; ++r )
	{
		RankBuf &rb = ranks[r];
		SparkKvShard sh = SparkK3KvShardContext(r, XR_RANKS);
		rb.mla_pool = (uint8_t *)XrDevice((uint64_t)pages * mla_page_bytes);
		rb.idx_pool = (uint8_t *)XrDevice((uint64_t)pages * idx_page_bytes);
		rb.mla_err = (LmKvAccessError *)XrDevice(sizeof(LmKvAccessError));
		rb.idx_err = (LmKvAccessError *)XrDevice(sizeof(LmKvAccessError));
		rb.sel_err = (LmFrameError *)XrDevice(sizeof(LmFrameError));
		XrCheck(LmKvShardViewInitialize<K3GlobalKv>(&rb.mla_view, rb.mla_pool, d_table, pages, 1u, pages, rb.mla_err, sh) == 0, "mla shard view");
		XrCheck(LmKvShardViewInitialize<K3IndexKv>(&rb.idx_view, rb.idx_pool, d_table, pages, 1u, pages, rb.idx_err, sh) == 0, "idx shard view");
		rb.partials_send = (float *)XrDevice((uint64_t)XR_RANKS * partial_stride * 4u);
		rb.partials_recv = (float *)XrDevice((uint64_t)XR_RANKS * partial_stride * 4u);
		rb.out_bf16 = (uint16_t *)XrDevice((uint64_t)rows * heads_per_rank * K3_KV_LORA_RANK * 2u);
		rb.scores = (float *)XrDevice((uint64_t)rows * index_stride * 4u);
		rb.idx_send = (uint2 *)XrDevice((uint64_t)rows * index_keep * sizeof(uint2));
	}
	uint2 *d_idx_gathered = (uint2 *)XrDevice((uint64_t)XR_RANKS * rows * index_keep * sizeof(uint2));
	uint32_t *d_bitmaps = (uint32_t *)XrDevice((uint64_t)rows * 2u * bitmap_words * 4u);
	uint32_t *d_selected = (uint32_t *)XrDevice((uint64_t)rows * K3_INDEX_TOPK * 4u);
	float *d_scalar_send = (float *)XrDevice((uint64_t)XR_RANKS * partial_stride * 4u);
	const uint64_t split_floats = 3u * LmLatentShardDecodeSplitFloats(1u, K3_MLA_HEADS, K3_KV_LORA_RANK);
	float *d_split_buf = (float *)XrDevice(split_floats * 4u);
	uint16_t *d_seq_q = (uint16_t *)XrDevice((uint64_t)XR_RANKS * SparkK3KvShardQueryStride(1u, XR_RANKS) * 2u);

	auto load_prefix = [&]()
	{
		for ( uint32_t r = 0u; r < XR_RANKS; ++r )
		{
			cudaMemset(ranks[r].mla_pool, 0, (size_t)pages * mla_page_bytes);
			cudaMemset(ranks[r].idx_pool, 0, (size_t)pages * idx_page_bytes);
			LM_LAUNCH((LmKvShardStoreKernel<K3GlobalKv, K3_LAYER_THREADS>), anchor, K3_LAYER_THREADS, 0, 0,
				ranks[r].mla_view, d_prefix_mla, d_prefix_seq, d_prefix_pos, anchor, slot_elems);
			if ( indexed != 0u )
				LM_LAUNCH((LmKvShardStoreKernel<K3IndexKv, K3_LAYER_THREADS>), anchor, K3_LAYER_THREADS, 0, 0,
					ranks[r].idx_view, d_prefix_idx, d_prefix_seq, d_prefix_pos, anchor, K3_INDEX_DIM);
		}
		XrCheck(cudaDeviceSynchronize() == cudaSuccess, "prefix store sync");
	};
	auto all_to_all_partials = [&](uint64_t p_stride)
	{
		for ( uint32_t dst = 0u; dst < XR_RANKS; ++dst )
			for ( uint32_t src = 0u; src < XR_RANKS; ++src )
				cudaMemcpy(ranks[dst].partials_recv + (uint64_t)src * p_stride,
					ranks[src].partials_send + (uint64_t)dst * p_stride,
					(size_t)p_stride * sizeof(float), cudaMemcpyDeviceToDevice);
	};

	load_prefix();
	cudaMemcpy(d_ctx, &layout.context, 4u, cudaMemcpyHostToDevice);
	for ( uint32_t r = 0u; r < XR_RANKS; ++r )
	{
		LM_LAUNCH((LmKvShardStoreKernel<K3GlobalKv, K3_LAYER_THREADS>), rows, K3_LAYER_THREADS, 0, 0,
			ranks[r].mla_view, d_tree_mla, d_seq, d_slots, rows, slot_elems);
		if ( indexed != 0u )
			LM_LAUNCH((LmKvShardStoreKernel<K3IndexKv, K3_LAYER_THREADS>), rows, K3_LAYER_THREADS, 0, 0,
				ranks[r].idx_view, d_tree_idx, d_seq, d_slots, rows, K3_INDEX_DIM);
	}
	std::vector<uint32_t> tree_selected((size_t)rows * K3_INDEX_TOPK, 0u);
	if ( indexed != 0u )
	{
		for ( uint32_t r = 0u; r < XR_RANKS; ++r )
		{
			XrCheck((LmWeightedSparseScoreLaunch<K3IndexKv, K3_INDEX_HEADS, K3_INDEX_DIM, LmKvShardView>(
				d_idx_q, d_idx_w, ranks[r].idx_view, d_seq, d_ctx, d_pos, rows, index_stride, index_scale, ranks[r].scores, 0, d_forks, d_offsets)) == cudaSuccess,
				"tree index score");
			XrCheck(LmIndexSelectLocalLaunch<LM_INDEX_SELECT_THREADS>(ranks[r].idx_view.shard, ranks[r].scores, index_stride,
				d_seq, d_ctx, d_pos, index_keep, K3_INDEX_TOPK, rows, ranks[r].idx_send, ranks[r].sel_err, 0) == cudaSuccess, "tree index local select");
			cudaMemcpy(d_idx_gathered + (uint64_t)r * rows * index_keep, ranks[r].idx_send, (size_t)rows * index_keep * sizeof(uint2), cudaMemcpyDeviceToDevice);
		}
		XrCheck(LmIndexSelectMergeLaunch<LM_INDEX_SELECT_THREADS>(d_idx_gathered, (uint64_t)rows * index_keep, ranks[0].idx_view.shard, index_keep,
			d_seq, d_ctx, d_pos, K3_INDEX_TOPK, K3_INDEX_TOPK, rows, bitmap_words, d_bitmaps, d_selected, ranks[0].sel_err, 0) == cudaSuccess, "tree index merge");
		XrCheck(cudaDeviceSynchronize() == cudaSuccess, "tree index sync");
		cudaMemcpy(tree_selected.data(), d_selected, tree_selected.size() * 4u, cudaMemcpyDeviceToHost);
	}
	std::vector<float> tree_scalar_r0((size_t)XR_RANKS * partial_stride);
	for ( uint32_t r = 0u; r < XR_RANKS; ++r )
	{
		if ( indexed != 0u )
			XrCheck((LmLatentShardAttendLaunch<K3GlobalKv, LmKvShardView, K3_KV_LORA_RANK, K3_QK_UNROTATED_DIM>(
				ranks[r].mla_view, d_query_all, query_stride, heads_per_rank, d_seq, d_ctx, d_pos, d_selected, K3_INDEX_TOPK, K3_INDEX_TOPK,
				K3_MLA_QK_SCALE, ranks[r].partials_send, partial_stride, rows, 0, d_forks, d_offsets)) == cudaSuccess, "tree sparse attend");
		else
			XrCheck((LmLatentShardSplitLaunch<K3GlobalKv, LmKvShardView, K3_KV_LORA_RANK, K3_QK_UNROTATED_DIM>(
				ranks[r].mla_view, d_query_all, query_stride, heads_per_rank, d_seq, d_ctx, d_pos, K3_MLA_QK_SCALE,
				d_split_buf, split_floats, ranks[r].partials_send, partial_stride, rows,
				LmLatentShardDecodeSplits(1u), 0, d_forks, d_offsets)) == cudaSuccess, "tree dense split");
		if ( r == 0u )
		{
			XrCheck((LmLatentShardPartialLaunch<K3GlobalKv, LmKvShardView, K3_KV_LORA_RANK, K3_QK_UNROTATED_DIM>(
				ranks[0].mla_view, d_query_all, query_stride, heads_per_rank, d_seq, d_ctx, d_pos,
				indexed != 0u ? d_selected : 0, indexed != 0u ? K3_INDEX_TOPK : 0u, indexed != 0u ? K3_INDEX_TOPK : 0u,
				K3_MLA_QK_SCALE, d_scalar_send, partial_stride, rows, 0, d_forks, d_offsets)) == cudaSuccess, "tree scalar partial");
			cudaMemcpy(tree_scalar_r0.data(), d_scalar_send, tree_scalar_r0.size() * 4u, cudaMemcpyDeviceToHost);
		}
	}
	all_to_all_partials(partial_stride);
	std::vector<std::vector<uint16_t>> tree_mla_out(XR_RANKS, std::vector<uint16_t>((size_t)rows * heads_per_rank * K3_KV_LORA_RANK));
	for ( uint32_t r = 0u; r < XR_RANKS; ++r )
	{
		XrCheck(LmLatentShardMergeLaunch<K3_KV_LORA_RANK>(ranks[r].partials_recv, partial_stride, XR_RANKS, heads_per_rank, ranks[r].out_bf16, rows, 0) == cudaSuccess, "tree mla merge");
		XrCheck(cudaDeviceSynchronize() == cudaSuccess, "tree mla sync");
		cudaMemcpy(tree_mla_out[r].data(), ranks[r].out_bf16, tree_mla_out[r].size() * 2u, cudaMemcpyDeviceToHost);
	}

	const uint64_t q1_stride = SparkK3KvShardQueryStride(1u, XR_RANKS);
	const uint64_t p1_stride = SparkK3KvShardPartialStride(1u, XR_RANKS);
	auto run_seq_mla = [&](uint32_t src_row, uint32_t pos_val, const uint16_t *q_host_flat,
		std::vector<std::vector<uint16_t>> &out_ranks, std::vector<uint32_t> *sel_out, std::vector<float> *scalar_r0_out)
	{
		const uint32_t ctx_val = pos_val + 1u;
		cudaMemcpy(d_ctx, &ctx_val, 4u, cudaMemcpyHostToDevice);
		cudaMemcpy(d_pos, &pos_val, 4u, cudaMemcpyHostToDevice);
		std::vector<uint16_t> q1_packed((size_t)XR_RANKS * q1_stride, 0u);
		for ( uint32_t r = 0u; r < XR_RANKS; ++r )
			memcpy(q1_packed.data() + (size_t)r * q1_stride,
				q_host_flat + (size_t)r * heads_per_rank * K3_MLA_KV_A_DIM,
				(size_t)heads_per_rank * K3_MLA_KV_A_DIM * 2u);
		cudaMemcpy(d_seq_q, q1_packed.data(), q1_packed.size() * 2u, cudaMemcpyHostToDevice);
		for ( uint32_t r = 0u; r < XR_RANKS; ++r )
		{
			LM_LAUNCH((LmKvShardStoreKernel<K3GlobalKv, K3_LAYER_THREADS>), 1u, K3_LAYER_THREADS, 0, 0,
				ranks[r].mla_view, d_tree_mla + (size_t)src_row * slot_elems, d_seq, d_pos, 1u, slot_elems);
			if ( indexed != 0u )
				LM_LAUNCH((LmKvShardStoreKernel<K3IndexKv, K3_LAYER_THREADS>), 1u, K3_LAYER_THREADS, 0, 0,
					ranks[r].idx_view, d_tree_idx + (size_t)src_row * K3_INDEX_DIM, d_seq, d_pos, 1u, K3_INDEX_DIM);
		}
		if ( indexed != 0u )
		{
			for ( uint32_t r = 0u; r < XR_RANKS; ++r )
			{
				XrCheck((LmWeightedSparseScoreLaunch<K3IndexKv, K3_INDEX_HEADS, K3_INDEX_DIM, LmKvShardView>(
					d_idx_q + (size_t)src_row * K3_INDEX_QUERY, d_idx_w + (size_t)src_row * K3_INDEX_HEADS,
					ranks[r].idx_view, d_seq, d_ctx, d_pos, 1u, index_stride, index_scale, ranks[r].scores, 0)) == cudaSuccess, "seq index score");
				XrCheck(LmIndexSelectLocalLaunch<LM_INDEX_SELECT_THREADS>(ranks[r].idx_view.shard, ranks[r].scores, index_stride,
					d_seq, d_ctx, d_pos, index_keep, K3_INDEX_TOPK, 1u, ranks[r].idx_send, ranks[r].sel_err, 0) == cudaSuccess, "seq index local");
				cudaMemcpy(d_idx_gathered + (uint64_t)r * index_keep, ranks[r].idx_send, (size_t)index_keep * sizeof(uint2), cudaMemcpyDeviceToDevice);
			}
			XrCheck(LmIndexSelectMergeLaunch<LM_INDEX_SELECT_THREADS>(d_idx_gathered, (uint64_t)index_keep, ranks[0].idx_view.shard, index_keep,
				d_seq, d_ctx, d_pos, K3_INDEX_TOPK, K3_INDEX_TOPK, 1u, bitmap_words, d_bitmaps, d_selected, ranks[0].sel_err, 0) == cudaSuccess, "seq index merge");
			if ( sel_out != 0 )
			{
				sel_out->resize(K3_INDEX_TOPK);
				cudaMemcpy(sel_out->data(), d_selected, (size_t)K3_INDEX_TOPK * 4u, cudaMemcpyDeviceToHost);
			}
		}
		for ( uint32_t r = 0u; r < XR_RANKS; ++r )
		{
			if ( indexed != 0u )
				XrCheck((LmLatentShardAttendLaunch<K3GlobalKv, LmKvShardView, K3_KV_LORA_RANK, K3_QK_UNROTATED_DIM>(
					ranks[r].mla_view, d_seq_q, q1_stride, heads_per_rank, d_seq, d_ctx, d_pos, d_selected, K3_INDEX_TOPK, K3_INDEX_TOPK,
					K3_MLA_QK_SCALE, ranks[r].partials_send, p1_stride, 1u, 0)) == cudaSuccess, "seq sparse attend");
			else
				XrCheck((LmLatentShardSplitLaunch<K3GlobalKv, LmKvShardView, K3_KV_LORA_RANK, K3_QK_UNROTATED_DIM>(
					ranks[r].mla_view, d_seq_q, q1_stride, heads_per_rank, d_seq, d_ctx, d_pos, K3_MLA_QK_SCALE,
					d_split_buf, split_floats, ranks[r].partials_send, p1_stride, 1u,
					LmLatentShardDecodeSplits(1u), 0)) == cudaSuccess, "seq dense split");
			if ( r == 0u && scalar_r0_out != 0 )
			{
				XrCheck((LmLatentShardPartialLaunch<K3GlobalKv, LmKvShardView, K3_KV_LORA_RANK, K3_QK_UNROTATED_DIM>(
					ranks[0].mla_view, d_seq_q, q1_stride, heads_per_rank, d_seq, d_ctx, d_pos,
					indexed != 0u ? d_selected : 0, indexed != 0u ? K3_INDEX_TOPK : 0u, indexed != 0u ? K3_INDEX_TOPK : 0u,
					K3_MLA_QK_SCALE, d_scalar_send, p1_stride, 1u, 0)) == cudaSuccess, "seq scalar partial");
				scalar_r0_out->resize((size_t)XR_RANKS * p1_stride);
				cudaMemcpy(scalar_r0_out->data(), d_scalar_send, scalar_r0_out->size() * 4u, cudaMemcpyDeviceToHost);
			}
		}
		all_to_all_partials(p1_stride);
		out_ranks.assign(XR_RANKS, std::vector<uint16_t>((size_t)heads_per_rank * K3_KV_LORA_RANK));
		for ( uint32_t r = 0u; r < XR_RANKS; ++r )
		{
			XrCheck(LmLatentShardMergeLaunch<K3_KV_LORA_RANK>(ranks[r].partials_recv, p1_stride, XR_RANKS, heads_per_rank, ranks[r].out_bf16, 1u, 0) == cudaSuccess, "seq mla merge");
			XrCheck(cudaDeviceSynchronize() == cudaSuccess, "seq mla sync");
			cudaMemcpy(out_ranks[r].data(), ranks[r].out_bf16, out_ranks[r].size() * 2u, cudaMemcpyDeviceToHost);
		}
	};

	const std::vector<std::vector<uint32_t>> paths = {
		{ 0u, 1u, 2u, 3u, 4u },
		{ 0u, 5u, 6u, 7u },
		{ 0u, 1u, 2u, 8u, 9u, 10u }
	};
	const uint64_t rec_floats = (uint64_t)heads_per_rank * LM_LATENT_SHARD_RECORD_FLOATS(K3_KV_LORA_RANK);
	uint32_t mla_diffs = 0u, sel_diffs = 0u, scalar_diffs = 0u;
	for ( uint32_t p = 0u; p < paths.size(); ++p )
	{
		load_prefix();
		for ( uint32_t step = 0u; step < paths[p].size(); ++step )
		{
			const uint32_t row_id = paths[p][step];
			std::vector<uint16_t> q_row((size_t)XR_RANKS * heads_per_rank * K3_MLA_KV_A_DIM);
			for ( uint32_t r = 0u; r < XR_RANKS; ++r )
				memcpy(q_row.data() + (size_t)r * heads_per_rank * K3_MLA_KV_A_DIM,
					h_query_all.data() + (size_t)r * query_stride + (size_t)row_id * heads_per_rank * K3_MLA_KV_A_DIM,
					(size_t)heads_per_rank * K3_MLA_KV_A_DIM * 2u);
			std::vector<std::vector<uint16_t>> seq_out;
			std::vector<uint32_t> seq_sel;
			std::vector<float> seq_scalar;
			run_seq_mla(row_id, anchor + step, q_row.data(), seq_out, indexed != 0u ? &seq_sel : 0, &seq_scalar);
			if ( indexed != 0u && memcmp(tree_selected.data() + (size_t)row_id * K3_INDEX_TOPK, seq_sel.data(), K3_INDEX_TOPK * 4u) != 0 )
				sel_diffs++;
			for ( uint32_t r = 0u; r < XR_RANKS; ++r )
			{
				if ( memcmp(tree_mla_out[r].data() + (size_t)row_id * heads_per_rank * K3_KV_LORA_RANK, seq_out[r].data(), seq_out[r].size() * 2u) != 0 )
					mla_diffs++;
				if ( memcmp(tree_scalar_r0.data() + (size_t)r * partial_stride + (size_t)row_id * rec_floats,
						seq_scalar.data() + (size_t)r * p1_stride, (size_t)rec_floats * 4u) != 0 )
					scalar_diffs++;
			}
		}
	}
	XrCheck(mla_diffs == 0u && sel_diffs == 0u && scalar_diffs == 0u,
		indexed != 0u ? "MLA+DSA sparse tree verify matches sequential decode bit-for-bit across all 16 ranks, all 11 tree rows, top-K indices and scalar partials"
			: "MLA dense split-K tree verify matches sequential decode bit-for-bit across all 16 ranks, all 11 tree rows and scalar partials");

	const std::vector<std::vector<uint32_t>> accept_paths = {
		{ 0u, 1u, 2u, 3u },
		{ 0u, 1u, 2u, 8u, 9u, 10u }
	};
	for ( uint32_t a = 0u; a < accept_paths.size(); ++a )
	{
		const auto &ap = accept_paths[a];
		const uint32_t kept = (uint32_t)ap.size();
		load_prefix();
		for ( uint32_t r = 0u; r < XR_RANKS; ++r )
		{
			LM_LAUNCH((LmKvShardStoreKernel<K3GlobalKv, K3_LAYER_THREADS>), rows, K3_LAYER_THREADS, 0, 0,
				ranks[r].mla_view, d_tree_mla, d_seq, d_slots, rows, slot_elems);
			if ( indexed != 0u )
				LM_LAUNCH((LmKvShardStoreKernel<K3IndexKv, K3_LAYER_THREADS>), rows, K3_LAYER_THREADS, 0, 0,
					ranks[r].idx_view, d_tree_idx, d_seq, d_slots, rows, K3_INDEX_DIM);
		}
		cudaMemcpy(d_path, ap.data(), kept * 4u, cudaMemcpyHostToDevice);
		cudaMemcpy(d_kept, &kept, 4u, cudaMemcpyHostToDevice);
		for ( uint32_t r = 0u; r < XR_RANKS; ++r )
		{
			LM_LAUNCH((LmKvShardPathMoveKernel<K3GlobalKv, K3_LAYER_THREADS>), rows, K3_LAYER_THREADS, 0, 0,
				ranks[r].mla_view, d_seq, d_path, d_kept, d_slots, slot_elems);
			if ( indexed != 0u )
				LM_LAUNCH((LmKvShardPathMoveKernel<K3IndexKv, K3_LAYER_THREADS>), rows, K3_LAYER_THREADS, 0, 0,
					ranks[r].idx_view, d_seq, d_path, d_kept, d_slots, K3_INDEX_DIM);
		}
		XrCheck(cudaDeviceSynchronize() == cudaSuccess, "MLA PathMove sync");
		std::vector<std::vector<uint16_t>> keep_n1, keep_n2, seq_n1, seq_n2, seq_tmp;
		run_seq_mla(11u, anchor + kept, h_cont_query.data(), keep_n1, 0, 0);
		run_seq_mla(12u, anchor + kept + 1u, h_cont_query.data() + (size_t)XR_RANKS * heads_per_rank * K3_MLA_KV_A_DIM, keep_n2, 0, 0);
		load_prefix();
		for ( uint32_t step = 0u; step < kept; ++step )
		{
			const uint32_t row_id = ap[step];
			std::vector<uint16_t> q_row((size_t)XR_RANKS * heads_per_rank * K3_MLA_KV_A_DIM);
			for ( uint32_t r = 0u; r < XR_RANKS; ++r )
				memcpy(q_row.data() + (size_t)r * heads_per_rank * K3_MLA_KV_A_DIM,
					h_query_all.data() + (size_t)r * query_stride + (size_t)row_id * heads_per_rank * K3_MLA_KV_A_DIM,
					(size_t)heads_per_rank * K3_MLA_KV_A_DIM * 2u);
			run_seq_mla(row_id, anchor + step, q_row.data(), seq_tmp, 0, 0);
		}
		run_seq_mla(11u, anchor + kept, h_cont_query.data(), seq_n1, 0, 0);
		run_seq_mla(12u, anchor + kept + 1u, h_cont_query.data() + (size_t)XR_RANKS * heads_per_rank * K3_MLA_KV_A_DIM, seq_n2, 0, 0);
		uint32_t post_diffs = 0u;
		for ( uint32_t r = 0u; r < XR_RANKS; ++r )
			if ( memcmp(keep_n1[r].data(), seq_n1[r].data(), seq_n1[r].size() * 2u) != 0 ||
				memcmp(keep_n2[r].data(), seq_n2[r].data(), seq_n2[r].size() * 2u) != 0 )
				post_diffs++;
		XrCheck(post_diffs == 0u, a == 0u ? "MLA KeepPath trunk + 2 post-KeepPath steps match sequential decode bit-for-bit"
			: "MLA KeepPath off-trunk branch + 2 post-KeepPath steps match sequential decode bit-for-bit");
	}
	printf("TREE-MLA(%s anchor=%u): 16 ranks, 11 tree rows, KeepPath (trunk & off-trunk) + 2 post-KeepPath steps: 0 bit diffs\n",
		indexed != 0u ? "DSA-sparse" : "dense-split16", anchor);

	for ( uint32_t r = 0u; r < XR_RANKS; ++r )
	{
		cudaFree(ranks[r].mla_pool); cudaFree(ranks[r].idx_pool);
		cudaFree(ranks[r].mla_err); cudaFree(ranks[r].idx_err); cudaFree(ranks[r].sel_err);
		cudaFree(ranks[r].partials_send); cudaFree(ranks[r].partials_recv);
		cudaFree(ranks[r].out_bf16); cudaFree(ranks[r].scores); cudaFree(ranks[r].idx_send);
	}
	cudaFree(d_table); cudaFree(d_prefix_mla); cudaFree(d_prefix_idx);
	cudaFree(d_tree_mla); cudaFree(d_tree_idx); cudaFree(d_query_all);
	cudaFree(d_idx_q); cudaFree(d_idx_w); cudaFree(d_prefix_seq); cudaFree(d_prefix_pos);
	cudaFree(d_seq); cudaFree(d_pos); cudaFree(d_slots); cudaFree(d_forks); cudaFree(d_offsets);
	cudaFree(d_ctx); cudaFree(d_path); cudaFree(d_kept);
	cudaFree(d_idx_gathered); cudaFree(d_bitmaps); cudaFree(d_selected);
	cudaFree(d_scalar_send); cudaFree(d_split_buf); cudaFree(d_seq_q);
}

static void XrTreeMlaTp1Verify(void)
{
	const uint32_t anchor = 90u, rows = 11u, total_rows = 13u, heads = 8u;
	const uint32_t slot_elems = K3GlobalKv::kSlotBytes / 2u;
	SparkSpeculationTreeLayout layout;
	std::vector<uint32_t> parents, positions, slots, forks, offsets, loads, saves;
	XrBuildTreeSpec(anchor, 1u, &layout, parents, positions, slots, forks, offsets, loads, saves);
	const uint32_t pages = (uint32_t)K3GlobalKv::PagesForTokens(layout.context + 8u);
	std::vector<uint32_t> h_table(pages);
	for ( uint32_t p = 0u; p < pages; ++p ) h_table[p] = p;
	std::vector<uint16_t> h_prefix((size_t)anchor * slot_elems), h_tree((size_t)total_rows * slot_elems);
	std::vector<uint16_t> h_q((size_t)total_rows * heads * K3_MLA_KV_A_DIM);
	for ( auto &x : h_prefix ) x = XrRandBf16();
	for ( auto &x : h_tree ) x = XrRandBf16();
	for ( auto &x : h_q ) x = XrRandBf16();
	uint8_t *d_pool = (uint8_t *)XrDevice((uint64_t)pages * K3GlobalKv::kPageBytes);
	uint32_t *d_table = (uint32_t *)XrDevice((uint64_t)pages * 4u);
	LmKvAccessError *d_err = (LmKvAccessError *)XrDevice(sizeof(LmKvAccessError));
	cudaMemcpy(d_table, h_table.data(), (size_t)pages * 4u, cudaMemcpyHostToDevice);
	LmKvView kv;
	XrCheck(LmKvViewInitialize(&kv, d_pool, d_table, pages, 1u, pages, d_err) == 0, "tp1 kv view");
	LmKvShardView shard_kv = { kv, SparkK3KvShardContext(0u, 1u) };
	uint16_t *d_prefix = (uint16_t *)XrDevice(h_prefix.size() * 2u), *d_tree = (uint16_t *)XrDevice(h_tree.size() * 2u);
	uint16_t *d_q = (uint16_t *)XrDevice(h_q.size() * 2u), *d_out = (uint16_t *)XrDevice((size_t)rows * heads * K3_KV_LORA_RANK * 2u);
	cudaMemcpy(d_prefix, h_prefix.data(), h_prefix.size() * 2u, cudaMemcpyHostToDevice);
	cudaMemcpy(d_tree, h_tree.data(), h_tree.size() * 2u, cudaMemcpyHostToDevice);
	cudaMemcpy(d_q, h_q.data(), h_q.size() * 2u, cudaMemcpyHostToDevice);
	std::vector<uint32_t> h_pseq(anchor, 0u), h_ppos(anchor), h_seq(rows, 0u);
	for ( uint32_t i = 0u; i < anchor; ++i ) h_ppos[i] = i;
	uint32_t *d_pseq = (uint32_t *)XrDevice((uint64_t)anchor * 4u), *d_ppos = (uint32_t *)XrDevice((uint64_t)anchor * 4u);
	uint32_t *d_seq = (uint32_t *)XrDevice((uint64_t)rows * 4u), *d_pos = (uint32_t *)XrDevice((uint64_t)rows * 4u);
	uint32_t *d_slots = (uint32_t *)XrDevice((uint64_t)rows * 4u), *d_forks = (uint32_t *)XrDevice((uint64_t)rows * 4u);
	uint32_t *d_offsets = (uint32_t *)XrDevice((uint64_t)rows * 4u), *d_ctx = (uint32_t *)XrDevice(4u);
	uint32_t *d_path = (uint32_t *)XrDevice((uint64_t)rows * 4u), *d_kept = (uint32_t *)XrDevice(4u);
	cudaMemcpy(d_pseq, h_pseq.data(), (size_t)anchor * 4u, cudaMemcpyHostToDevice);
	cudaMemcpy(d_ppos, h_ppos.data(), (size_t)anchor * 4u, cudaMemcpyHostToDevice);
	cudaMemcpy(d_seq, h_seq.data(), (size_t)rows * 4u, cudaMemcpyHostToDevice);
	cudaMemcpy(d_pos, positions.data(), (size_t)rows * 4u, cudaMemcpyHostToDevice);
	cudaMemcpy(d_slots, slots.data(), (size_t)rows * 4u, cudaMemcpyHostToDevice);
	cudaMemcpy(d_forks, forks.data(), (size_t)rows * 4u, cudaMemcpyHostToDevice);
	cudaMemcpy(d_offsets, offsets.data(), (size_t)rows * 4u, cudaMemcpyHostToDevice);
	cudaMemcpy(d_ctx, &layout.context, 4u, cudaMemcpyHostToDevice);
	LM_LAUNCH((LmKvStoreKernel<K3GlobalKv, K3_LAYER_THREADS>), anchor, K3_LAYER_THREADS, 0, 0, kv, d_prefix, d_pseq, d_ppos, anchor, slot_elems);
	LM_LAUNCH((LmKvStoreKernel<K3GlobalKv, K3_LAYER_THREADS>), rows, K3_LAYER_THREADS, 0, 0, kv, d_tree, d_seq, d_slots, rows, slot_elems);
	LM_LAUNCH((LmAttentionDecodeKernel<K3GlobalKv, K3_ATTN_THREADS, K3_KV_LORA_RANK, K3_QK_UNROTATED_DIM>), dim3(rows, heads), K3_ATTN_THREADS, 0, 0,
		d_q, d_q, kv, d_seq, d_ctx, 0, 0u, heads, K3_MLA_QK_SCALE, d_out, d_pos, d_forks, d_offsets);
	XrCheck(cudaDeviceSynchronize() == cudaSuccess, "tp1 tree attn sync");
	std::vector<uint16_t> tree_out((size_t)rows * heads * K3_KV_LORA_RANK), seq_out((size_t)heads * K3_KV_LORA_RANK);
	cudaMemcpy(tree_out.data(), d_out, tree_out.size() * 2u, cudaMemcpyDeviceToHost);
	const uint32_t branch2[6] = { 0u, 1u, 2u, 8u, 9u, 10u };
	const uint32_t kept = 6u;
	cudaMemcpy(d_path, branch2, kept * 4u, cudaMemcpyHostToDevice);
	cudaMemcpy(d_kept, &kept, 4u, cudaMemcpyHostToDevice);
	LM_LAUNCH((LmKvShardPathMoveKernel<K3GlobalKv, K3_LAYER_THREADS>), rows, K3_LAYER_THREADS, 0, 0, shard_kv, d_seq, d_path, d_kept, d_slots, slot_elems);
	const uint32_t next_pos = anchor + kept, next_ctx = next_pos + 1u;
	cudaMemcpy(d_pos, &next_pos, 4u, cudaMemcpyHostToDevice);
	cudaMemcpy(d_ctx, &next_ctx, 4u, cudaMemcpyHostToDevice);
	LM_LAUNCH((LmKvStoreKernel<K3GlobalKv, K3_LAYER_THREADS>), 1u, K3_LAYER_THREADS, 0, 0, kv, d_tree + (size_t)11u * slot_elems, d_seq, d_pos, 1u, slot_elems);
	LM_LAUNCH((LmAttentionDecodeKernel<K3GlobalKv, K3_ATTN_THREADS, K3_KV_LORA_RANK, K3_QK_UNROTATED_DIM>), dim3(1u, heads), K3_ATTN_THREADS, 0, 0,
		d_q + (size_t)11u * heads * K3_MLA_KV_A_DIM, d_q + (size_t)11u * heads * K3_MLA_KV_A_DIM, kv, d_seq, d_ctx, 0, 0u, heads, K3_MLA_QK_SCALE, d_out, d_pos);
	XrCheck(cudaDeviceSynchronize() == cudaSuccess, "tp1 post-keep sync");
	std::vector<uint16_t> keep_next((size_t)heads * K3_KV_LORA_RANK);
	cudaMemcpy(keep_next.data(), d_out, keep_next.size() * 2u, cudaMemcpyDeviceToHost);
	LM_LAUNCH((LmKvStoreKernel<K3GlobalKv, K3_LAYER_THREADS>), anchor, K3_LAYER_THREADS, 0, 0, kv, d_prefix, d_pseq, d_ppos, anchor, slot_elems);
	uint32_t tp1_diffs = 0u;
	for ( uint32_t step = 0u; step < kept; ++step )
	{
		const uint32_t p = anchor + step, c = p + 1u, row_id = branch2[step];
		cudaMemcpy(d_pos, &p, 4u, cudaMemcpyHostToDevice);
		cudaMemcpy(d_ctx, &c, 4u, cudaMemcpyHostToDevice);
		LM_LAUNCH((LmKvStoreKernel<K3GlobalKv, K3_LAYER_THREADS>), 1u, K3_LAYER_THREADS, 0, 0, kv, d_tree + (size_t)row_id * slot_elems, d_seq, d_pos, 1u, slot_elems);
		LM_LAUNCH((LmAttentionDecodeKernel<K3GlobalKv, K3_ATTN_THREADS, K3_KV_LORA_RANK, K3_QK_UNROTATED_DIM>), dim3(1u, heads), K3_ATTN_THREADS, 0, 0,
			d_q + (size_t)row_id * heads * K3_MLA_KV_A_DIM, d_q + (size_t)row_id * heads * K3_MLA_KV_A_DIM, kv, d_seq, d_ctx, 0, 0u, heads, K3_MLA_QK_SCALE, d_out, d_pos);
		XrCheck(cudaDeviceSynchronize() == cudaSuccess, "tp1 seq sync");
		cudaMemcpy(seq_out.data(), d_out, seq_out.size() * 2u, cudaMemcpyDeviceToHost);
		if ( memcmp(tree_out.data() + (size_t)row_id * heads * K3_KV_LORA_RANK, seq_out.data(), seq_out.size() * 2u) != 0 )
			tp1_diffs++;
	}
	cudaMemcpy(d_pos, &next_pos, 4u, cudaMemcpyHostToDevice);
	cudaMemcpy(d_ctx, &next_ctx, 4u, cudaMemcpyHostToDevice);
	LM_LAUNCH((LmKvStoreKernel<K3GlobalKv, K3_LAYER_THREADS>), 1u, K3_LAYER_THREADS, 0, 0, kv, d_tree + (size_t)11u * slot_elems, d_seq, d_pos, 1u, slot_elems);
	LM_LAUNCH((LmAttentionDecodeKernel<K3GlobalKv, K3_ATTN_THREADS, K3_KV_LORA_RANK, K3_QK_UNROTATED_DIM>), dim3(1u, heads), K3_ATTN_THREADS, 0, 0,
		d_q + (size_t)11u * heads * K3_MLA_KV_A_DIM, d_q + (size_t)11u * heads * K3_MLA_KV_A_DIM, kv, d_seq, d_ctx, 0, 0u, heads, K3_MLA_QK_SCALE, d_out, d_pos);
	XrCheck(cudaDeviceSynchronize() == cudaSuccess, "tp1 seq next sync");
	cudaMemcpy(seq_out.data(), d_out, seq_out.size() * 2u, cudaMemcpyDeviceToHost);
	XrCheck(tp1_diffs == 0u && memcmp(keep_next.data(), seq_out.data(), seq_out.size() * 2u) == 0,
		"TP1 MLA tree verify and off-trunk KeepPath match sequential decode bit-for-bit");
	printf("TREE-MLA(TP1): 11 rows + off-trunk KeepPath + post-KeepPath step: 0 bit diffs\n");
	cudaFree(d_pool); cudaFree(d_table); cudaFree(d_err); cudaFree(d_prefix); cudaFree(d_tree);
	cudaFree(d_q); cudaFree(d_out); cudaFree(d_pseq); cudaFree(d_ppos); cudaFree(d_seq);
	cudaFree(d_pos); cudaFree(d_slots); cudaFree(d_forks); cudaFree(d_offsets); cudaFree(d_ctx);
	cudaFree(d_path); cudaFree(d_kept);
}

static void XrTreeGraphVerify(void)
{
	SparkStageRunnerStep step;
	memset(&step, 0, sizeof(step));
	step.sequences = 1u;
	step.commit = 0u;
	step.tree = 1u;
	for ( uint32_t r : { 2u, 11u, 16u, 24u, 32u } )
	{
		step.rows = r;
		XrCheck(K3ModelGraphKey(0, &step) == 2u, "K3ModelGraphKey returns 2 for tree verify frames up to 32 rows (runner maps key 2 + tree=1 to slot 3)");
	}
	step.commit = 1u;
	step.tree = 0u;
	step.rows = 1u;
	XrCheck(K3ModelGraphKey(0, &step) == 1u, "K3ModelGraphKey returns 1 for committed decode");

	const uint32_t rows = 11u, anchor = 120u, heads = K3_KDA_HEADS / XR_RANKS;
	const uint32_t qk = heads * K3_KDA_KEY_DIM, vdim = heads * K3_KDA_VALUE_DIM;
	const uint32_t state_bytes = (uint32_t)K3_KDA_RANK_STATE_SLOT_BYTES(heads, 0u);
	const uint32_t slot_elems = K3GlobalKv::kSlotBytes / 2u;
	SparkSpeculationTreeLayout layout;
	std::vector<uint32_t> parents, positions, slots, forks, offsets, loads, saves;
	XrBuildTreeSpec(anchor, XR_RANKS, &layout, parents, positions, slots, forks, offsets, loads, saves);
	const uint32_t pages = (uint32_t)K3GlobalKv::PagesForTokens(layout.context + 8u);
	const uint64_t mla_page_bytes = LmKvShardPageStride<K3GlobalKv>(SparkK3KvShardContext(0u, XR_RANKS));
	const uint64_t q_stride = SparkK3KvShardQueryStride(rows, XR_RANKS), p_stride = SparkK3KvShardPartialStride(rows, XR_RANKS);
	uint8_t *d_state = (uint8_t *)XrDevice(state_bytes), *d_pool = (uint8_t *)XrDevice((uint64_t)pages * mla_page_bytes);
	uint16_t *d_wq_win = (uint16_t *)XrDevice((size_t)qk * K3_KDA_CONV_KERNEL * 2u);
	uint16_t *d_wk_win = (uint16_t *)XrDevice((size_t)qk * K3_KDA_CONV_KERNEL * 2u);
	uint16_t *d_wv_win = (uint16_t *)XrDevice((size_t)vdim * K3_KDA_CONV_KERNEL * 2u);
	float *d_cw = (float *)XrDevice((size_t)qk * K3_KDA_CONV_KERNEL * 4u);
	uint16_t *d_rq = (uint16_t *)XrDevice((size_t)rows * qk * 2u), *d_rk = (uint16_t *)XrDevice((size_t)rows * qk * 2u), *d_rv = (uint16_t *)XrDevice((size_t)rows * vdim * 2u);
	uint16_t *d_q = (uint16_t *)XrDevice((size_t)rows * qk * 2u), *d_k = (uint16_t *)XrDevice((size_t)rows * qk * 2u), *d_v = (uint16_t *)XrDevice((size_t)rows * vdim * 2u);
	uint16_t *d_out = (uint16_t *)XrDevice((size_t)rows * vdim * 2u);
	float *d_ret = (float *)XrDevice((size_t)rows * qk * 4u), *d_gate = (float *)XrDevice((size_t)rows * heads * 4u);
	float *d_snap = (float *)XrDevice(LmDeltaRuleSnapshotBytes(heads, K3_KDA_KEY_DIM, K3_KDA_VALUE_DIM));
	uint16_t *d_mla_slot = (uint16_t *)XrDevice((size_t)rows * slot_elems * 2u);
	uint16_t *d_mla_q = (uint16_t *)XrDevice((size_t)XR_RANKS * q_stride * 2u);
	const uint64_t graph_split_floats = 3u * LmLatentShardDecodeSplitFloats(1u, K3_MLA_HEADS, K3_KV_LORA_RANK);
	float *d_psend = (float *)XrDevice((size_t)XR_RANKS * p_stride * 4u), *d_precv = (float *)XrDevice(graph_split_floats * 4u);
	uint32_t *d_table = (uint32_t *)XrDevice((size_t)pages * 4u), *d_idx = (uint32_t *)XrDevice(4u), *d_begin = (uint32_t *)XrDevice(8u);
	uint32_t *d_order = (uint32_t *)XrDevice((size_t)rows * 4u), *d_seq = (uint32_t *)XrDevice((size_t)rows * 4u);
	uint32_t *d_pos = (uint32_t *)XrDevice((size_t)rows * 4u), *d_slots = (uint32_t *)XrDevice((size_t)rows * 4u);
	uint32_t *d_forks = (uint32_t *)XrDevice((size_t)rows * 4u), *d_offsets = (uint32_t *)XrDevice((size_t)rows * 4u);
	uint32_t *d_parent = (uint32_t *)XrDevice((size_t)rows * 4u), *d_loads = (uint32_t *)XrDevice((size_t)rows * 4u), *d_saves = (uint32_t *)XrDevice((size_t)rows * 4u);
	uint32_t *d_ctx = (uint32_t *)XrDevice(4u);
	LmKvAccessError *d_err = (LmKvAccessError *)XrDevice(sizeof(LmKvAccessError));
	std::vector<uint32_t> h_table(pages), h_order(rows);
	for ( uint32_t i = 0u; i < pages; ++i ) h_table[i] = i;
	for ( uint32_t i = 0u; i < rows; ++i ) h_order[i] = i;
	const uint32_t h_begin[2] = { 0u, rows };
	cudaMemcpy(d_table, h_table.data(), (size_t)pages * 4u, cudaMemcpyHostToDevice);
	cudaMemcpy(d_begin, h_begin, 8u, cudaMemcpyHostToDevice);
	cudaMemcpy(d_order, h_order.data(), (size_t)rows * 4u, cudaMemcpyHostToDevice);
	cudaMemcpy(d_pos, positions.data(), (size_t)rows * 4u, cudaMemcpyHostToDevice);
	cudaMemcpy(d_slots, slots.data(), (size_t)rows * 4u, cudaMemcpyHostToDevice);
	cudaMemcpy(d_forks, forks.data(), (size_t)rows * 4u, cudaMemcpyHostToDevice);
	cudaMemcpy(d_offsets, offsets.data(), (size_t)rows * 4u, cudaMemcpyHostToDevice);
	cudaMemcpy(d_parent, parents.data(), (size_t)rows * 4u, cudaMemcpyHostToDevice);
	cudaMemcpy(d_loads, loads.data(), (size_t)rows * 4u, cudaMemcpyHostToDevice);
	cudaMemcpy(d_saves, saves.data(), (size_t)rows * 4u, cudaMemcpyHostToDevice);
	cudaMemcpy(d_ctx, &layout.context, 4u, cudaMemcpyHostToDevice);
	LmKvShardView mla_view;
	XrCheck(LmKvShardViewInitialize<K3GlobalKv>(&mla_view, d_pool, d_table, pages, 1u, pages, d_err, SparkK3KvShardContext(0u, XR_RANKS)) == 0, "graph mla view");
	auto fill_inputs = [&](uint32_t seed_add)
	{
		xr_rng += seed_add;
		std::vector<uint16_t> buf((size_t)rows * qk);
		for ( auto &x : buf ) x = XrRandBf16();
		cudaMemcpy(d_rq, buf.data(), buf.size() * 2u, cudaMemcpyHostToDevice);
		for ( auto &x : buf ) x = XrRandBf16();
		cudaMemcpy(d_rk, buf.data(), buf.size() * 2u, cudaMemcpyHostToDevice);
		for ( auto &x : buf ) x = XrRandBf16();
		cudaMemcpy(d_rv, buf.data(), buf.size() * 2u, cudaMemcpyHostToDevice);
		std::vector<float> rbuf((size_t)rows * qk, 0.85f), gbuf((size_t)rows * heads, 0.5f), cwbuf((size_t)qk * K3_KDA_CONV_KERNEL, 0.25f);
		cudaMemcpy(d_ret, rbuf.data(), rbuf.size() * 4u, cudaMemcpyHostToDevice);
		cudaMemcpy(d_gate, gbuf.data(), gbuf.size() * 4u, cudaMemcpyHostToDevice);
		cudaMemcpy(d_cw, cwbuf.data(), cwbuf.size() * 4u, cudaMemcpyHostToDevice);
		std::vector<uint16_t> sbuf((size_t)rows * slot_elems);
		for ( auto &x : sbuf ) x = XrRandBf16();
		cudaMemcpy(d_mla_slot, sbuf.data(), sbuf.size() * 2u, cudaMemcpyHostToDevice);
		std::vector<uint16_t> qbuf((size_t)XR_RANKS * q_stride);
		for ( auto &x : qbuf ) x = XrRandBf16();
		cudaMemcpy(d_mla_q, qbuf.data(), qbuf.size() * 2u, cudaMemcpyHostToDevice);
	};
	auto enqueue_verify = [&](uint32_t tree_mode, cudaStream_t s)
	{
		const LmCausalConvStreams<float> conv = {
			{ d_wq_win, d_wk_win, d_wv_win }, { d_rq, d_rk, d_rv }, { d_cw, d_cw, d_cw }, { d_q, d_k, d_v }, { qk, qk, vdim }
		};
		XrCheck(LmCausalConvStreamsLaunch<K3_KDA_CONV_KERNEL, LM_CONV_SWISH, float>(conv, d_idx, d_begin, 0, 1u, 0u, d_order, rows, s, tree_mode != 0u ? d_parent : 0) == cudaSuccess, "graph conv");
		LM_LAUNCH((LmL2NormalisePerHeadKernel<K3_LAYER_THREADS, K3_KDA_KEY_DIM>), dim3(rows, heads), K3_LAYER_THREADS, 0, s, d_q, heads, rows, K3_RMS_EPSILON);
		LM_LAUNCH((LmL2NormalisePerHeadKernel<K3_LAYER_THREADS, K3_KDA_KEY_DIM>), dim3(rows, heads), K3_LAYER_THREADS, 0, s, d_k, heads, rows, K3_RMS_EPSILON);
		XrCheck(LmDeltaRuleLaunch<K3_LAYER_THREADS, K3_KDA_KEY_DIM, K3_KDA_VALUE_DIM, float>(rows, d_state, state_bytes, d_idx, d_begin, 0,
			d_q, d_k, d_v, d_ret, d_gate, d_out, heads, 1u, 1u, 0u, d_order, s, tree_mode != 0u ? d_loads : 0, tree_mode != 0u ? d_saves : 0, tree_mode != 0u ? d_snap : 0) == cudaSuccess, "graph delta");
		LM_LAUNCH((LmKvShardStoreKernel<K3GlobalKv, K3_LAYER_THREADS>), rows, K3_LAYER_THREADS, 0, s,
			mla_view, d_mla_slot, d_seq, tree_mode != 0u ? d_slots : d_pos, rows, slot_elems);
		XrCheck((LmLatentShardSplitLaunch<K3GlobalKv, LmKvShardView, K3_KV_LORA_RANK, K3_QK_UNROTATED_DIM>(
			mla_view, d_mla_q, q_stride, heads, d_seq, d_ctx, d_pos, K3_MLA_QK_SCALE, d_precv, graph_split_floats,
			d_psend, p_stride, rows, LmLatentShardDecodeSplits(1u), s, tree_mode != 0u ? d_forks : 0, tree_mode != 0u ? d_offsets : 0)) == cudaSuccess, "graph mla split");
	};
	cudaStream_t stream = 0;
	cudaGraph_t tree_graph = 0, chain_graph = 0;
	cudaGraphExec_t tree_exec = 0, chain_exec = 0;
	XrCheck(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) == cudaSuccess, "graph stream");
	fill_inputs(1u);
	enqueue_verify(1u, stream);
	enqueue_verify(0u, stream);
	XrCheck(cudaStreamSynchronize(stream) == cudaSuccess, "graph warmup");
	XrCheck(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal) == cudaSuccess, "begin tree capture");
	enqueue_verify(1u, stream);
	XrCheck(cudaStreamEndCapture(stream, &tree_graph) == cudaSuccess && cudaGraphInstantiate(&tree_exec, tree_graph, 0) == cudaSuccess, "instantiate tree graph");
	XrCheck(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal) == cudaSuccess, "begin chain capture");
	enqueue_verify(0u, stream);
	XrCheck(cudaStreamEndCapture(stream, &chain_graph) == cudaSuccess && cudaGraphInstantiate(&chain_exec, chain_graph, 0) == cudaSuccess, "instantiate chain graph");
	for ( uint32_t iter = 0u; iter < 2u; ++iter )
	{
		fill_inputs(100u + iter * 17u);
		std::vector<uint16_t> g_kda((size_t)rows * vdim), e_kda((size_t)rows * vdim);
		std::vector<float> g_mla((size_t)XR_RANKS * p_stride), e_mla((size_t)XR_RANKS * p_stride);
		XrCheck(cudaGraphLaunch(tree_exec, stream) == cudaSuccess && cudaStreamSynchronize(stream) == cudaSuccess, "launch tree graph");
		cudaMemcpy(g_kda.data(), d_out, g_kda.size() * 2u, cudaMemcpyDeviceToHost);
		cudaMemcpy(g_mla.data(), d_psend, g_mla.size() * 4u, cudaMemcpyDeviceToHost);
		enqueue_verify(1u, stream);
		XrCheck(cudaStreamSynchronize(stream) == cudaSuccess, "eager tree sync");
		cudaMemcpy(e_kda.data(), d_out, e_kda.size() * 2u, cudaMemcpyDeviceToHost);
		cudaMemcpy(e_mla.data(), d_psend, e_mla.size() * 4u, cudaMemcpyDeviceToHost);
		XrCheck(memcmp(g_kda.data(), e_kda.data(), g_kda.size() * 2u) == 0 && memcmp(g_mla.data(), e_mla.data(), g_mla.size() * 4u) == 0,
			"tree graph replay matches eager execution bit-for-bit");
		XrCheck(cudaGraphLaunch(chain_exec, stream) == cudaSuccess && cudaStreamSynchronize(stream) == cudaSuccess, "launch chain graph");
		cudaMemcpy(g_kda.data(), d_out, g_kda.size() * 2u, cudaMemcpyDeviceToHost);
		cudaMemcpy(g_mla.data(), d_psend, g_mla.size() * 4u, cudaMemcpyDeviceToHost);
		enqueue_verify(0u, stream);
		XrCheck(cudaStreamSynchronize(stream) == cudaSuccess, "eager chain sync");
		cudaMemcpy(e_kda.data(), d_out, e_kda.size() * 2u, cudaMemcpyDeviceToHost);
		cudaMemcpy(e_mla.data(), d_psend, e_mla.size() * 4u, cudaMemcpyDeviceToHost);
		XrCheck(memcmp(g_kda.data(), e_kda.data(), g_kda.size() * 2u) == 0 && memcmp(g_mla.data(), e_mla.data(), g_mla.size() * 4u) == 0,
			"chain graph replay matches eager execution bit-for-bit after alternating with tree graph");
	}
	printf("TREE-GRAPH: captured tree=1 and tree=0 CUDA graphs, alternated 2x with 0 recaptures and 0 bit diffs\n");
	cudaGraphExecDestroy(tree_exec); cudaGraphExecDestroy(chain_exec);
	cudaGraphDestroy(tree_graph); cudaGraphDestroy(chain_graph);
	cudaStreamDestroy(stream);
	cudaFree(d_state); cudaFree(d_pool); cudaFree(d_wq_win); cudaFree(d_wk_win); cudaFree(d_wv_win);
	cudaFree(d_cw); cudaFree(d_rq); cudaFree(d_rk); cudaFree(d_rv);
	cudaFree(d_q); cudaFree(d_k); cudaFree(d_v); cudaFree(d_out);
	cudaFree(d_ret); cudaFree(d_gate); cudaFree(d_snap);
	cudaFree(d_mla_slot); cudaFree(d_mla_q); cudaFree(d_psend); cudaFree(d_precv);
	cudaFree(d_table); cudaFree(d_idx); cudaFree(d_begin); cudaFree(d_order);
	cudaFree(d_seq); cudaFree(d_pos); cudaFree(d_slots); cudaFree(d_forks); cudaFree(d_offsets);
	cudaFree(d_parent); cudaFree(d_loads); cudaFree(d_saves); cudaFree(d_ctx); cudaFree(d_err);
}

static float XrBf16ToF32(uint16_t bits)
{
	uint32_t u = (uint32_t)bits << 16u;
	float f = 0.0f;
	memcpy(&f, &u, sizeof(f));
	return f;
}

static uint16_t XrF32ToBf16(float value)
{
	uint32_t u = 0u;
	memcpy(&u, &value, sizeof(u));
	u += 0x7fffu + ((u >> 16u) & 1u);
	return (uint16_t)(u >> 16u);
}

static void XrDirectGatherSparseParityVerify(void)
{
	const uint32_t rows = 32u, heads = K3_MLA_HEADS / XR_RANKS, total_ctx = 2080u;
	const uint32_t blocks = (rows + 7u) / 8u;
	constexpr uint32_t UCAP = 8u * K3_INDEX_TOPK;
	constexpr uint32_t SHARED_BYTES = LmPrefillAttnSharedBytes<K3_MLA_KV_A_DIM, LM_PREFILL_ATTN_QUERIES>();
	cudaStream_t stream = 0;
	XrCheck(cudaStreamCreate(&stream) == cudaSuccess, "direct gather parity stream");
	std::vector<uint16_t> q_h((size_t)rows * heads * K3_MLA_KV_A_DIM);
	std::vector<uint16_t> kv_h((size_t)total_ctx * K3_MLA_KV_A_DIM);
	for ( size_t i = 0; i < q_h.size(); ++i )
	{
		uint32_t x = (uint32_t)(i * 2654435761u) ^ 0x13579bdfu;
		x ^= x >> 13u; x *= 0x5bd1e995u; x ^= x >> 15u;
		q_h[i] = XrF32ToBf16(0.12f * (((float)(x & 0xffffu) - 32768.0f) / 32768.0f));
	}
	for ( size_t i = 0; i < kv_h.size(); ++i )
	{
		uint32_t x = (uint32_t)(i * 2246822519u) ^ 0x2468ace0u;
		x ^= x >> 13u; x *= 0x5bd1e995u; x ^= x >> 15u;
		kv_h[i] = XrF32ToBf16(0.12f * (((float)(x & 0xffffu) - 32768.0f) / 32768.0f));
	}
	std::vector<uint32_t> pos_h(rows), sel_h((size_t)rows * K3_INDEX_TOPK, LM_INDEX_SHARD_NO_POSITION);
	for ( uint32_t r = 0u; r < rows; ++r )
	{
		pos_h[r] = total_ctx - rows + r;
		std::vector<uint8_t> seen(total_ctx, 0u);
		uint32_t w = 0u;
		for ( uint32_t k = 0u; k < K3_INDEX_TOPK; ++k )
		{
			const uint32_t v = (k + r * 7u) % (pos_h[r] + 1u);
			if ( seen[v] == 0u ) { seen[v] = 1u; sel_h[(size_t)r * K3_INDEX_TOPK + w++] = v; }
		}
		std::sort(sel_h.begin() + (size_t)r * K3_INDEX_TOPK, sel_h.begin() + (size_t)r * K3_INDEX_TOPK + w);
	}
	uint16_t *q_d = (uint16_t *)XrDevice(q_h.size() * 2u);
	uint16_t *kv_d = (uint16_t *)XrDevice(kv_h.size() * 2u);
	uint16_t *out_d = (uint16_t *)XrDevice((size_t)rows * heads * K3_KV_LORA_RANK * 2u);
	uint32_t *seq_d = (uint32_t *)XrDevice(rows * 4u);
	uint32_t *ctx_d = (uint32_t *)XrDevice(4u);
	uint32_t *pos_d = (uint32_t *)XrDevice(rows * 4u);
	uint32_t *sel_d = (uint32_t *)XrDevice(sel_h.size() * 4u);
	uint32_t *upos_d = (uint32_t *)XrDevice((size_t)blocks * UCAP * 4u);
	uint8_t *umask_d = (uint8_t *)XrDevice((size_t)blocks * UCAP);
	uint32_t *ucnt_d = (uint32_t *)XrDevice(blocks * 4u);
	cudaMemcpy(q_d, q_h.data(), q_h.size() * 2u, cudaMemcpyHostToDevice);
	cudaMemcpy(kv_d, kv_h.data(), kv_h.size() * 2u, cudaMemcpyHostToDevice);
	cudaMemcpy(pos_d, pos_h.data(), rows * 4u, cudaMemcpyHostToDevice);
	cudaMemcpy(ctx_d, &total_ctx, 4u, cudaMemcpyHostToDevice);
	cudaMemcpy(sel_d, sel_h.data(), sel_h.size() * 4u, cudaMemcpyHostToDevice);
	LmPrefillSparseUnionKernel<8u, K3_INDEX_TOPK><<<blocks, LM_PREFILL_UNION_THREADS, 0, stream>>>(
		sel_d, pos_d, rows, (const uint32_t *)0, UCAP, upos_d, umask_d, ucnt_d);
	XrCheck(K3Skl2Launch(K3SKL2_k3_mla_gather_prefill_sparse_256, dim3(blocks, heads / 2u), LM_PREFILL_ATTN_THREADS, SHARED_BYTES, stream,
		q_d, kv_d, seq_d, ctx_d, pos_d, (float)K3_MLA_QK_SCALE, out_d, rows, heads, total_ctx, upos_d, umask_d, ucnt_d, UCAP) == SPARK_GPU_SUCCESS,
		"direct gather sparse skl2 launch");
	XrCheck(cudaStreamSynchronize(stream) == cudaSuccess, "direct gather sparse sync");
	std::vector<uint16_t> out_h((size_t)rows * heads * K3_KV_LORA_RANK);
	cudaMemcpy(out_h.data(), out_d, out_h.size() * 2u, cudaMemcpyDeviceToHost);
	double dot = 0.0, na = 0.0, nb = 0.0;
	const uint32_t check_rows[] = {0u, 7u, 15u, 31u};
	for ( uint32_t ci = 0u; ci < 4u; ++ci )
	{
		const uint32_t r = check_rows[ci], lim = pos_h[r];
		std::vector<uint32_t> keys;
		for ( uint32_t k = 0u; k < K3_INDEX_TOPK; ++k )
		{
			const uint32_t p = sel_h[(size_t)r * K3_INDEX_TOPK + k];
			if ( p <= lim && p < total_ctx ) keys.push_back(p);
		}
		for ( uint32_t h = 0u; h < heads; ++h )
		{
			const uint16_t *qp = q_h.data() + ((size_t)r * heads + h) * K3_MLA_KV_A_DIM;
			std::vector<float> sc(keys.size());
			float mx = -1e30f;
			for ( size_t i = 0; i < keys.size(); ++i )
			{
				const uint16_t *kp = kv_h.data() + (size_t)keys[i] * K3_MLA_KV_A_DIM;
				float s = 0.0f;
				for ( uint32_t d = 0u; d < K3_MLA_KV_A_DIM; ++d ) s += XrBf16ToF32(qp[d]) * XrBf16ToF32(kp[d]);
				s *= K3_MLA_QK_SCALE;
				sc[i] = s;
				if ( s > mx ) mx = s;
			}
			float sum = 0.0f;
			for ( size_t i = 0; i < keys.size(); ++i ) { sc[i] = expf(sc[i] - mx); sum += sc[i]; }
			const uint16_t *got = out_h.data() + ((size_t)r * heads + h) * K3_KV_LORA_RANK;
			for ( uint32_t d = 0u; d < K3_KV_LORA_RANK; ++d )
			{
				float ref = 0.0f;
				for ( size_t i = 0; i < keys.size(); ++i ) ref += sc[i] * XrBf16ToF32(kv_h[(size_t)keys[i] * K3_MLA_KV_A_DIM + d]);
				ref /= sum;
				const float g = XrBf16ToF32(got[d]);
				dot += (double)ref * g; na += (double)ref * ref; nb += (double)g * g;
			}
		}
	}
	const double cos = dot / sqrt(na * nb);
	printf("DIRECT-GATHER-SPARSE: cos=%.6f\n", cos);
	XrCheck(cos >= 0.9999, "K3MlaGatherPrefillKernel<true> matches FP32 sparse reference");
	cudaStreamDestroy(stream);
	cudaFree(q_d); cudaFree(kv_d); cudaFree(out_d); cudaFree(seq_d); cudaFree(ctx_d);
	cudaFree(pos_d); cudaFree(sel_d); cudaFree(upos_d); cudaFree(umask_d); cudaFree(ucnt_d);
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
	XrTreeKdaVerify();
	XrTreeMlaDsaVerify(120u, 0u);
	XrTreeMlaDsaVerify(2400u, 1u);
	XrTreeMlaTp1Verify();
	XrTreeGraphVerify();
	XrDirectGatherSparseParityVerify();
	if ( xr_failures != 0 )
	{
		printf("FAIL k3_index_exchange_cuda: %d failures\n", xr_failures);
		return 1;
	}
	printf("PASS k3_index_exchange_cuda: chunks with a row per rank exchange candidates by row owner, merge only their own rows and gather the selection; "
		"every rank ends with the gathered path's selection, ties included; K3 KDA, MLA dense, MLA+DSA sparse, TP1 MLA, KeepPath and alternating tree/chain CUDA graphs match sequential decode bit-for-bit\n");
	return 0;
}
