#pragma once

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <fcntl.h>
#include <unistd.h>

#include "sparkpipe/spark_k3_bind.h"
#include "sparkpipe/spark_pack_window.h"
#include "sparkpipe/spark_stage_runner.h"
#include "sparkpipe/spark_stage_runner_model.h"
#include "sparkpipe/spark_weight_stream.h"
#include "sparkpipe/spark_weightd_lazy_pack.h"
#include "sparkpipe/spark_weightd_map.h"
#include "inference/llms/kimi_k3/streamed.cuh"
#include "spark_k3_weight_binds.cuh"

#define K3_STREAMED_RANGE_GAP 4096u
#define K3_STREAMED_SHARDS 16u
#define K3_STREAMED_HEADS (K3_MLA_HEADS / K3_STREAMED_SHARDS)
#define K3_STREAMED_ROW_ALIGN 256u
#define K3_STREAMED_CHUNKS 8u
#define K3_STREAMED_STREAM_CHUNK (16u << 20)
#define K3_STREAMED_STREAM_DEPTH 16u
#define K3_STREAMED_MAX_ALLOCATIONS 64u
#define K3_STREAMED_GATHER_BYTES 4096u

typedef struct K3StreamedRoundContext
{
	const SparkStageRunnerServices *services;
	SparkStatus status;
} K3StreamedRoundContext;

static void K3StreamedRound(void *context, void *stream, uint32_t operation, uint32_t rows, uint32_t row_elements, const void *send, void *receive)
{
	K3StreamedRoundContext *round = (K3StreamedRoundContext *)context;
	SparkStatus status;
	if ( round == 0 || round->services == 0 || round->services->round == 0 )
		return;
	status = round->services->round(round->services->context, stream, operation, SPARK_STAGE_RUNNER_ROUND_ALL, rows, row_elements, send, receive);
	if ( status != SPARK_STATUS_OK && round->status == SPARK_STATUS_OK )
		round->status = status;
}

static int32_t K3StreamedLayerRanges(uint64_t payload_base, const SparkK3BoundLayer *bound, SparkPackWindowLayer *ranges)
{
	uint64_t start[SPARK_K3_BIND_MAX_NAMES], end[SPARK_K3_BIND_MAX_NAMES], expert_start = UINT64_MAX, expert_end = 0u;
	uint32_t count = 0u, index, other, dense;
	if ( bound == 0 || ranges == 0 || bound->tensor_count > SPARK_K3_BIND_MAX_NAMES )
		return -1;
	memset(ranges, 0, sizeof(*ranges));
	for ( index = 0u; index < bound->tensor_count; index++ )
	{
		const SparkK3PackEntry *entry = &bound->tensors[index].entry;
		const uint64_t first = payload_base + entry->payload_offset, last = first + entry->bytes;
		const size_t length = strlen(bound->tensors[index].name);
		if ( length >= 16u && (strcmp(bound->tensors[index].name + length - 16u, "expert_w1_weight") == 0 ||
			strcmp(bound->tensors[index].name + length - 16u, "expert_w2_weight") == 0) )
		{
			expert_start = first < expert_start ? first : expert_start;
			expert_end = last > expert_end ? last : expert_end;
			continue;
		}
		start[count] = first;
		end[count] = last;
		count++;
	}
	for ( index = 1u; index < count; index++ )
		for ( other = index; other > 0u && start[other - 1u] > start[other]; other-- )
		{
			const uint64_t swap_start = start[other], swap_end = end[other];
			start[other] = start[other - 1u];
			end[other] = end[other - 1u];
			start[other - 1u] = swap_start;
			end[other - 1u] = swap_end;
		}
	if ( expert_end != 0u )
	{
		ranges->experts.offset = expert_start;
		ranges->experts.bytes = expert_end - expert_start;
	}
	for ( index = 0u, dense = 0u; index < count; index++ )
	{
		const uint32_t crosses = expert_end != 0u && dense != 0u &&
			ranges->dense[dense - 1u].offset < expert_start && start[index] >= expert_end ? 1u : 0u;
		if ( expert_end != 0u && start[index] < expert_end && end[index] > expert_start )
			return -1;
		if ( dense != 0u && crosses == 0u && start[index] <= ranges->dense[dense - 1u].offset + ranges->dense[dense - 1u].bytes + K3_STREAMED_RANGE_GAP )
		{
			const uint64_t range_end = ranges->dense[dense - 1u].offset + ranges->dense[dense - 1u].bytes;
			if ( end[index] > range_end )
				ranges->dense[dense - 1u].bytes = end[index] - ranges->dense[dense - 1u].offset;
			continue;
		}
		if ( dense == SPARK_PACK_WINDOW_MAX_DENSE_RANGES )
			return -1;
		ranges->dense[dense].offset = start[index];
		ranges->dense[dense].bytes = end[index] - start[index];
		dense++;
	}
	ranges->dense_count = dense;
	return dense != 0u ? 0 : -1;
}

struct K3Streamed
{
	SparkStageRunnerStreamedPrefillConfig config;
	SparkK3Dispatch *dispatch;
	const SparkK3ModuleState *module;
	SparkWeightdLazyPack *lazy_pack;
	uint32_t rank;
	uint32_t node_rows;
	uint32_t vocab_rows;
	uint32_t tag;
	uint32_t opened;
	uint64_t key_capacity;
	int partner_fd;
	const uint8_t *pool;
	uint8_t *region;
	uint64_t region_bytes;
	uint64_t engine_bytes;
	SparkWeightStream *stream;
	SparkPackWindow window;
	SparkPackWindowLayer ranges[K3_LAYERS];
	const SparkK3BoundLayer *bound[K3_LAYERS];
	SparkPackWindowLayer vocab;
	K3LayerWeights shards[K3_STREAMED_SHARDS];
	K3LayerBuffers b;
	K3StreamedJoin join;
	uint16_t *partial;
	float *hidden_accumulator;
	float *latent_accumulator;
	K3StreamedKdaExchange kda;
	K3StreamedKdaState states[K3_KDA_LAYER_COUNT * K3_STREAMED_SHARDS];
	K3StreamedMlaExchange mla;
	uint32_t *tokens_device;
	uint32_t *positions_device;
	uint32_t *context_device;
	uint32_t *older_context_device;
	uint32_t *store_sequence_device;
	uint8_t *gather_device;
	uint32_t *host_staging;
	uint8_t *mla_keys;
	cudaStream_t gather_stream;
	cudaEvent_t slot_done[2];
	cudaEvent_t layer_done;
	K3StreamedRoundContext round;
	void *allocations[K3_STREAMED_MAX_ALLOCATIONS];
	uint32_t allocation_count;
};

template<class T>
static T *K3StreamedAlloc(K3Streamed *s, uint64_t count)
{
	void *pointer = 0;
	const uint64_t bytes = count * sizeof(T) + 256u;
	if ( s->allocation_count >= K3_STREAMED_MAX_ALLOCATIONS || cudaMalloc(&pointer, bytes) != cudaSuccess )
		return 0;
	if ( cudaMemset(pointer, 0, bytes) != cudaSuccess )
	{
		cudaFree(pointer);
		return 0;
	}
	s->allocations[s->allocation_count++] = pointer;
	s->engine_bytes += bytes;
	return (T *)pointer;
}

static SparkStatus K3StreamedReadOwn(void *context, uint64_t pack_offset, uint64_t bytes, uint8_t *destination)
{
	const K3Streamed *s = (const K3Streamed *)context;
	if ( s->pool == 0 || pack_offset + bytes > s->module->pack.file_bytes )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	memcpy(destination, s->pool + pack_offset, (size_t)bytes);
	return SPARK_STATUS_OK;
}

static SparkStatus K3StreamedReadPartner(void *context, uint64_t pack_offset, uint64_t bytes, uint8_t *destination)
{
	const K3Streamed *s = (const K3Streamed *)context;
	uint64_t done = 0u;
	while ( done < bytes )
	{
		const ssize_t got = pread(s->partner_fd, destination + done, (size_t)(bytes - done < (1ull << 30) ? bytes - done : (1ull << 30)), (off_t)(pack_offset + done));
		if ( got <= 0 )
			SPARK_FAIL(SPARK_STATUS_IO_ERROR);
		done += (uint64_t)got;
	}
	(void)posix_fadvise(s->partner_fd, (off_t)pack_offset, (off_t)bytes, POSIX_FADV_DONTNEED);
	return SPARK_STATUS_OK;
}

static SparkStatus K3StreamedGather(void *context, const void *mine, void *all, uint64_t bytes_per_rank)
{
	K3Streamed *s = (K3Streamed *)context;
	uint8_t *send = s->gather_device, *receive = s->gather_device + K3_STREAMED_GATHER_BYTES;
	if ( bytes_per_rank == 0u || bytes_per_rank > K3_STREAMED_GATHER_BYTES || (bytes_per_rank % 2u) != 0u )
		SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
	if ( cudaMemcpyAsync(send, mine, (size_t)bytes_per_rank, cudaMemcpyHostToDevice, s->gather_stream) != cudaSuccess )
		SPARK_FAIL(SPARK_STATUS_IO_ERROR);
	K3StreamedRound(&s->round, (void *)s->gather_stream, SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_GATHER, 1u, (uint32_t)(bytes_per_rank / 2u), send, receive);
	if ( s->round.status != SPARK_STATUS_OK ||
		cudaMemcpyAsync(all, receive, (size_t)(bytes_per_rank * K3_STREAMED_SHARDS), cudaMemcpyDeviceToHost, s->gather_stream) != cudaSuccess ||
		cudaStreamSynchronize(s->gather_stream) != cudaSuccess )
		SPARK_FAIL(s->round.status != SPARK_STATUS_OK ? s->round.status : SPARK_STATUS_IO_ERROR);
	return SPARK_STATUS_OK;
}

static void K3StreamedDestroy(K3Streamed *s)
{
	uint32_t index;
	if ( s == 0 )
		return;
	if ( s->stream != 0 )
		SparkWeightStreamClose(s->stream);
	if ( s->region != 0 )
	{
		(void)cudaHostUnregister(s->region);
		free(s->region);
	}
	if ( s->partner_fd >= 0 )
		close(s->partner_fd);
	for (index = 0u; index < 2u; index++)
		if ( s->slot_done[index] != 0 )
			(void)cudaEventDestroy(s->slot_done[index]);
	if ( s->layer_done != 0 )
		(void)cudaEventDestroy(s->layer_done);
	if ( s->gather_stream != 0 )
		(void)cudaStreamDestroy(s->gather_stream);
	cudaFree(s->mla_keys);
	cudaFreeHost(s->host_staging);
	for (index = 0u; index < s->allocation_count; index++)
		cudaFree(s->allocations[index]);
	delete s;
}

static SparkStatus K3StreamedBuffers(K3Streamed *s)
{
	const SparkK3ScratchWidths *w = &s->dispatch->widths;
	const K3LayerBuffers *own = s->dispatch->buffers_host;
	const uint64_t rows = s->node_rows, pass = (uint64_t)K3_STREAMED_SHARDS * rows;
	const uint64_t heads = own->kda_heads_rank, slot_floats = heads * K3_KDA_KEY_DIM * K3_KDA_VALUE_DIM;
	const uint64_t tail = (2u * heads * K3_KDA_KEY_DIM + heads * K3_KDA_VALUE_DIM) * K3_KDA_CONV_KERNEL;
	const uint64_t join_columns = K3_MLA_HEADS * K3_V_HEAD_DIM > K3_STREAMED_SHARDS * heads * K3_KDA_VALUE_DIM ? K3_MLA_HEADS * K3_V_HEAD_DIM : K3_STREAMED_SHARDS * heads * K3_KDA_VALUE_DIM;
	K3LayerBuffers *b = &s->b;
	uint32_t *chunk_host, chunk;
	uint8_t *state_memory;
	uint16_t *window_memory;
	memset(b, 0, sizeof(*b));
	b->kda_qkvb_rows = own->kda_qkvb_rows;
	b->kda_gate_rows = own->kda_gate_rows;
	b->kda_decay_up_rows = own->kda_decay_up_rows;
	b->kda_out_input = own->kda_out_input;
	b->mla_q_up_rows = own->mla_q_up_rows;
	b->mla_kv_b_value_rows = own->mla_kv_b_value_rows;
	b->mla_gate_rows = own->mla_gate_rows;
	b->mla_out_input = own->mla_out_input;
	b->routed_down_rows = own->routed_down_rows;
	b->routed_up_input = own->routed_up_input;
	b->expert_w1_output = own->expert_w1_output;
	b->expert_w2_input = own->expert_w2_input;
	b->shared_w1_rows = own->shared_w1_rows;
	b->shared_w2_input = own->shared_w2_input;
	b->dense_gate_up_rows = own->dense_gate_up_rows;
	b->dense_down_input = own->dense_down_input;
	b->kda_heads_rank = own->kda_heads_rank;
	b->mla_heads_rank = own->mla_heads_rank;
	b->tp_sharded = 1u;
	b->kv_shard.degree = 1u;
	b->kv_shard.rank = 0u;
	b->kv_shard.grain = 1u;
	b->cache_shard.pages = s->dispatch->mla_cache[0];
	b->cache_shard.shard = b->kv_shard;
	b->hidden_bf16 = K3StreamedAlloc<uint16_t>(s, rows * K3_HIDDEN);
	b->normed_bf16 = K3StreamedAlloc<uint16_t>(s, rows * K3_HIDDEN);
	b->fused_qkvb_bf16 = K3StreamedAlloc<uint16_t>(s, rows * w->qkvb);
	b->query_bf16 = K3StreamedAlloc<uint16_t>(s, rows * w->query);
	b->key_bf16 = K3StreamedAlloc<uint16_t>(s, rows * w->key);
	b->value_bf16 = K3StreamedAlloc<uint16_t>(s, rows * w->value);
	b->gate_bf16 = K3StreamedAlloc<uint16_t>(s, rows * w->gate);
	b->decay_logit_bf16 = K3StreamedAlloc<uint16_t>(s, rows * w->decay);
	b->latent_bf16 = K3StreamedAlloc<uint16_t>(s, rows * w->latent);
	b->kv_slot_bf16 = K3StreamedAlloc<uint16_t>(s, rows * K3_MLA_KV_A_DIM);
	b->attention_out_bf16 = K3StreamedAlloc<uint16_t>(s, rows * w->attention_out);
	b->shared_out_bf16 = K3StreamedAlloc<uint16_t>(s, rows * K3_HIDDEN);
	b->attnres_bank_bf16 = K3StreamedAlloc<uint16_t>(s, (uint64_t)K3_ATTNRES_MAX_SOURCES * rows * K3_HIDDEN);
	b->attnres_partial_bf16 = K3StreamedAlloc<uint16_t>(s, rows * K3_HIDDEN);
	b->kda_beta_logit = K3StreamedAlloc<uint16_t>(s, rows * K3_KDA_HEADS);
	b->kda_write_gate_out = K3StreamedAlloc<float>(s, rows * K3_KDA_HEADS);
	b->gate_up_bf16 = K3StreamedAlloc<uint16_t>(s, rows * w->gate_up);
	b->intermediate_bf16 = K3StreamedAlloc<uint16_t>(s, rows * w->intermediate);
	b->kda_retention = K3StreamedAlloc<float>(s, rows * w->retention);
	b->router_logits = K3StreamedAlloc<float>(s, rows * K3_EXPERTS);
	b->attnres_score_f32 = K3StreamedAlloc<float>(s, rows * K3_ATTNRES_MAX_SOURCES);
	b->latent_full_bf16 = K3StreamedAlloc<uint16_t>(s, rows * K3_ROUTED_EXPERT_HIDDEN);
	b->shared_mid_bf16 = K3StreamedAlloc<uint16_t>(s, rows * w->shared_mid);
	b->route_expert = K3StreamedAlloc<uint32_t>(s, rows * K3_TOP_K);
	b->route_packed_row = K3StreamedAlloc<uint32_t>(s, rows * K3_TOP_K);
	b->route_source_token = K3StreamedAlloc<uint32_t>(s, rows * K3_TOP_K);
	b->route_weight = K3StreamedAlloc<float>(s, rows * K3_TOP_K);
	b->group_row_offset = K3StreamedAlloc<uint32_t>(s, K3_EXPERTS + 1u);
	b->group_tile_prefix_w1 = K3StreamedAlloc<uint32_t>(s, K3_EXPERTS + 1u);
	b->group_tile_prefix_w2 = K3StreamedAlloc<uint32_t>(s, K3_EXPERTS + 1u);
	b->dense_row_offset = K3StreamedAlloc<uint32_t>(s, 2u);
	b->dense_tile_prefix = K3StreamedAlloc<uint32_t>(s, 2u);
	b->sequence_of_row = K3StreamedAlloc<uint32_t>(s, rows);
	b->positions = s->positions_device = K3StreamedAlloc<uint32_t>(s, rows);
	b->context_length = s->context_device = K3StreamedAlloc<uint32_t>(s, 1u);
	s->tokens_device = K3StreamedAlloc<uint32_t>(s, rows);
	s->partial = K3StreamedAlloc<uint16_t>(s, rows * K3_HIDDEN);
	s->hidden_accumulator = K3StreamedAlloc<float>(s, rows * K3_HIDDEN);
	s->latent_accumulator = K3StreamedAlloc<float>(s, rows * K3_ROUTED_EXPERT_HIDDEN);
	s->join.weight_capacity = (uint64_t)K3_HIDDEN * (join_columns > K3_ROUTED_EXPERT_HIDDEN ? join_columns : K3_ROUTED_EXPERT_HIDDEN);
	s->join.activation_capacity = rows * join_columns;
	s->join.weight = K3StreamedAlloc<uint16_t>(s, s->join.weight_capacity);
	s->join.activation = K3StreamedAlloc<uint16_t>(s, s->join.activation_capacity);
	s->gather_device = K3StreamedAlloc<uint8_t>(s, (uint64_t)(K3_STREAMED_SHARDS + 1u) * K3_STREAMED_GATHER_BYTES);
	memset(&s->kda, 0, sizeof(s->kda));
	s->kda.node = s->rank;
	s->kda.nodes = K3_STREAMED_SHARDS;
	s->kda.chunks = K3_STREAMED_CHUNKS;
	s->kda.chunk_row_begin = K3StreamedAlloc<uint32_t>(s, K3_STREAMED_CHUNKS + 1u);
	s->kda.chunk_slots = K3StreamedAlloc<uint32_t>(s, K3_STREAMED_CHUNKS);
	s->kda.tail_send = K3StreamedAlloc<uint16_t>(s, tail);
	s->kda.tails = K3StreamedAlloc<uint16_t>(s, (uint64_t)K3_STREAMED_SHARDS * tail);
	s->kda.summary_send = K3StreamedAlloc<float>(s, 2u * slot_floats);
	s->kda.summaries = K3StreamedAlloc<float>(s, (uint64_t)K3_STREAMED_SHARDS * 2u * slot_floats);
	s->kda.ends = K3StreamedAlloc<float>(s, K3_STREAMED_CHUNKS * slot_floats);
	s->kda.transforms = K3StreamedAlloc<float>(s, K3_STREAMED_CHUNKS * slot_floats);
	s->kda.incoming = K3StreamedAlloc<float>(s, K3_STREAMED_CHUNKS * slot_floats);
	s->kda.running = K3StreamedAlloc<float>(s, 2u * slot_floats);
	s->kda.q_windows = K3StreamedAlloc<uint16_t>(s, K3_STREAMED_CHUNKS * heads * K3_KDA_KEY_DIM * K3_KDA_CONV_KERNEL);
	s->kda.k_windows = K3StreamedAlloc<uint16_t>(s, K3_STREAMED_CHUNKS * heads * K3_KDA_KEY_DIM * K3_KDA_CONV_KERNEL);
	s->kda.v_windows = K3StreamedAlloc<uint16_t>(s, K3_STREAMED_CHUNKS * heads * K3_KDA_VALUE_DIM * K3_KDA_CONV_KERNEL);
	s->kda.zeros = K3StreamedAlloc<uint16_t>(s, rows * heads * K3_KDA_VALUE_DIM);
	s->kda.scratch = K3StreamedAlloc<uint16_t>(s, rows * heads * K3_KDA_VALUE_DIM);
	s->kda.collective = K3StreamedRound;
	s->kda.context = &s->round;
	state_memory = K3StreamedAlloc<uint8_t>(s, (uint64_t)K3_KDA_LAYER_COUNT * K3_STREAMED_SHARDS * slot_floats * sizeof(float));
	window_memory = K3StreamedAlloc<uint16_t>(s, (uint64_t)K3_KDA_LAYER_COUNT * K3_STREAMED_SHARDS * (2u * heads * K3_KDA_KEY_DIM + heads * K3_KDA_VALUE_DIM) * K3_KDA_CONV_KERNEL);
	memset(&s->mla, 0, sizeof(s->mla));
	s->mla.node = s->rank;
	s->mla.nodes = K3_STREAMED_SHARDS;
	s->mla.q_latents = K3StreamedAlloc<uint16_t>(s, pass * K3_Q_LORA_RANK);
	s->mla.query = K3StreamedAlloc<uint16_t>(s, pass * K3_STREAMED_HEADS * K3_MHA_QK);
	s->mla.partials = K3StreamedAlloc<float>(s, pass * LmGqaShardRecordFloats(K3_STREAMED_HEADS, K3_V_HEAD_DIM, 1u));
	s->mla.merged = K3StreamedAlloc<uint16_t>(s, pass * K3_STREAMED_HEADS * K3_V_HEAD_DIM);
	s->mla.value_send = K3StreamedAlloc<uint16_t>(s, pass * K3_STREAMED_HEADS * K3_V_HEAD_DIM);
	s->mla.values = K3StreamedAlloc<uint16_t>(s, pass * K3_STREAMED_HEADS * K3_V_HEAD_DIM);
	s->mla.pass_positions = K3StreamedAlloc<uint32_t>(s, pass);
	s->mla.pass_sequence_of_row = K3StreamedAlloc<uint32_t>(s, pass);
	s->mla.pass_row_offset = K3StreamedAlloc<uint32_t>(s, 2u);
	s->mla.pass_tile_prefix = K3StreamedAlloc<uint32_t>(s, 2u);
	s->mla.key_offset = K3StreamedAlloc<uint32_t>(s, s->dispatch->sequences);
	s->mla.key_context = K3StreamedAlloc<uint32_t>(s, 1u);
	s->mla.store_sequence_of_row = s->store_sequence_device = K3StreamedAlloc<uint32_t>(s, pass);
	s->mla.older_sequence = K3StreamedAlloc<uint32_t>(s, 1u);
	s->mla.older_offset = s->mla.key_offset;
	s->mla.older_context = s->older_context_device = K3StreamedAlloc<uint32_t>(s, s->dispatch->sequences);
	s->mla.collective = K3StreamedRound;
	s->mla.context = &s->round;
	if ( cudaMallocHost((void **)&s->host_staging, (size_t)(pass + K3_STREAMED_CHUNKS + 8u) * sizeof(uint32_t)) != cudaSuccess )
		s->host_staging = 0;
	if ( s->allocation_count == K3_STREAMED_MAX_ALLOCATIONS || s->host_staging == 0 || state_memory == 0 || window_memory == 0 || s->mla.older_context == 0 )
		SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
	for (uint32_t index = 0u; index < s->allocation_count; index++)
		if ( s->allocations[index] == 0 )
			SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
	for (uint32_t index = 0u; index < K3_KDA_LAYER_COUNT * K3_STREAMED_SHARDS; index++)
	{
		const uint64_t qk = heads * K3_KDA_KEY_DIM * K3_KDA_CONV_KERNEL, v = heads * K3_KDA_VALUE_DIM * K3_KDA_CONV_KERNEL;
		s->states[index].state_pool = state_memory + (uint64_t)index * slot_floats * sizeof(float);
		s->states[index].q_window = window_memory + (uint64_t)index * (2u * qk + v);
		s->states[index].k_window = s->states[index].q_window + qk;
		s->states[index].v_window = s->states[index].k_window + qk;
	}
	chunk_host = s->host_staging;
	for (chunk = 0u; chunk < K3_STREAMED_CHUNKS; chunk++)
		chunk_host[chunk] = chunk;
	if ( cudaMemcpy((void *)s->kda.chunk_slots, chunk_host, K3_STREAMED_CHUNKS * sizeof(uint32_t), cudaMemcpyHostToDevice) != cudaSuccess )
		SPARK_FAIL(SPARK_STATUS_IO_ERROR);
	return SPARK_STATUS_OK;
}

static SparkStatus K3StreamedCreate(const SparkStageRunnerStreamedPrefillConfig *config, SparkK3Dispatch *dispatch, const SparkK3ModuleState *module,
	SparkWeightdLazyPack *lazy_pack, uint32_t rank, uint64_t vocab_offset, uint64_t vocab_bytes, uint32_t vocab_rows, K3Streamed **out)
{
	K3Streamed *s;
	uint64_t dense_capacity = vocab_bytes, expert_capacity = 0u;
	uint32_t layer, index;
	int usable = 0;
	SparkStatus status;
	*out = 0;
	if ( config->node_rows == 0u || (config->node_rows % K3_STREAMED_ROW_ALIGN) != 0u || dispatch->tp_degree != K3_STREAMED_SHARDS ||
		dispatch->first_layer != 0u || dispatch->layer_count != K3_LAYERS || dispatch->kda_count != K3_KDA_LAYER_COUNT || dispatch->mla_count == 0u ||
		dispatch->buffers_host == 0 || dispatch->buffers_host->kv_shard.degree != K3_STREAMED_SHARDS || vocab_bytes == 0u || vocab_rows == 0u ||
		vocab_rows * K3_STREAMED_SHARDS != K3_VOCAB )
	{
		fprintf(stderr, "sparkpipe_k3: streamed prefill needs node rows in multiples of %u, all %u layers on %u ranks and a %u-row vocab slice per rank\n",
			K3_STREAMED_ROW_ALIGN, K3_LAYERS, K3_STREAMED_SHARDS, K3_VOCAB / K3_STREAMED_SHARDS);
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	}
	if ( cudaDeviceGetAttribute(&usable, cudaDevAttrCanUseHostPointerForRegisteredMem, dispatch->device) != cudaSuccess || usable == 0 )
	{
		fprintf(stderr, "sparkpipe_k3: streamed prefill needs a device that reads registered host memory at the host address\n");
		SPARK_FAIL(SPARK_STATUS_UNSUPPORTED);
	}
	s = new K3Streamed;
	memset(s, 0, sizeof(*s));
	s->partner_fd = -1;
	s->config = *config;
	s->dispatch = dispatch;
	s->module = module;
	s->lazy_pack = lazy_pack;
	s->rank = rank;
	s->node_rows = config->node_rows;
	s->vocab_rows = vocab_rows;
	s->vocab.dense_count = 1u;
	s->vocab.dense[0].offset = vocab_offset;
	s->vocab.dense[0].bytes = vocab_bytes;
	for (layer = 0u; layer < K3_LAYERS; layer++)
	{
		for (index = 0u; index < module->bound_count && s->bound[layer] == 0; index++)
			if ( module->bound[index].layer_index == layer )
				s->bound[layer] = &module->bound[index];
		if ( s->bound[layer] == 0 || K3StreamedLayerRanges(module->pack.payload_base, s->bound[layer], &s->ranges[layer]) != 0 ||
			(layer >= K3_FIRST_ROUTED_LAYER && (s->ranges[layer].experts.bytes == 0u || (s->ranges[layer].experts.bytes % 2u) != 0u)) )
		{
			fprintf(stderr, "sparkpipe_k3: streamed prefill cannot place layer %u of the pack in window ranges\n", layer);
			K3StreamedDestroy(s);
			SPARK_FAIL(SPARK_STATUS_PARSE_ERROR);
		}
		dense_capacity = SparkPackWindowDenseBytes(&s->ranges[layer]) > dense_capacity ? SparkPackWindowDenseBytes(&s->ranges[layer]) : dense_capacity;
		expert_capacity = s->ranges[layer].experts.bytes > expert_capacity ? s->ranges[layer].experts.bytes : expert_capacity;
	}
	s->region_bytes = SparkPackWindowRegionBytes(K3_STREAMED_SHARDS, 1u, dense_capacity, expert_capacity);
	s->region_bytes = (s->region_bytes + (2ull << 20) - 1u) & ~((2ull << 20) - 1u);
	s->region = (uint8_t *)aligned_alloc(2ull << 20, (size_t)s->region_bytes);
	if ( s->region == 0 || cudaHostRegister(s->region, (size_t)s->region_bytes, cudaHostRegisterDefault) != cudaSuccess )
	{
		fprintf(stderr, "sparkpipe_k3: the %llu-byte streamed weight window could not be allocated and registered\n", (unsigned long long)s->region_bytes);
		if ( s->region != 0 )
		{
			free(s->region);
			s->region = 0;
		}
		K3StreamedDestroy(s);
		SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
	}
	s->window.config.dense_capacity = dense_capacity;
	s->window.config.expert_capacity = expert_capacity;
	s->partner_fd = open(config->partner_pack_path, O_RDONLY | O_CLOEXEC);
	status = s->partner_fd >= 0 ? K3StreamedBuffers(s) : SPARK_STATUS_IO_ERROR;
	if ( status == SPARK_STATUS_OK && (cudaStreamCreateWithFlags(&s->gather_stream, cudaStreamNonBlocking) != cudaSuccess ||
		cudaEventCreateWithFlags(&s->slot_done[0], cudaEventDisableTiming) != cudaSuccess ||
		cudaEventCreateWithFlags(&s->slot_done[1], cudaEventDisableTiming) != cudaSuccess ||
		cudaEventCreateWithFlags(&s->layer_done, cudaEventDisableTiming) != cudaSuccess) )
		status = SPARK_STATUS_IO_ERROR;
	if ( status != SPARK_STATUS_OK )
	{
		fprintf(stderr, "sparkpipe_k3: streamed prefill resources failed status=%d (partner pack %s)\n", (int)status, config->partner_pack_path);
		K3StreamedDestroy(s);
		return status;
	}
	fprintf(stderr, "sparkpipe_k3: streamed prefill node_rows=%u window=%llu (dense %llu x %u ranks, experts %llu) engine=%llu partner=%s\n",
		s->node_rows, (unsigned long long)s->region_bytes, (unsigned long long)dense_capacity, K3_STREAMED_SHARDS, (unsigned long long)expert_capacity,
		(unsigned long long)s->engine_bytes, config->partner_pack_path);
	*out = s;
	return SPARK_STATUS_OK;
}

static SparkStatus K3StreamedOpen(K3Streamed *s, const SparkStageRunnerServices *services)
{
	SparkWeightStreamConfig stream_config;
	SparkPackWindowConfig window_config;
	const void *pool = 0;
	uint32_t resident = 0u;
	SparkStatus status;
	s->round.services = services;
	s->round.status = SPARK_STATUS_OK;
	if ( s->opened != 0u )
		return SPARK_STATUS_OK;
	status = s->lazy_pack != 0 && s->lazy_pack->map != 0 ? SparkWeightdMapResident(s->lazy_pack->map, SPARK_WEIGHTD_ATTACH_TIMEOUT_DEFAULT_NS, &resident) : SPARK_STATUS_INVALID_ARGUMENT;
	if ( status == SPARK_STATUS_OK && resident != 0u )
		status = SparkWeightdMapPool(s->lazy_pack->map, &pool);
	if ( status != SPARK_STATUS_OK || resident == 0u || pool == 0 )
	{
		fprintf(stderr, "sparkpipe_k3: streamed prefill reads the own shard from the resident pool, which is not present (status=%d resident=%u)\n", (int)status, resident);
		SPARK_FAIL(status != SPARK_STATUS_OK ? status : SPARK_STATUS_UNSUPPORTED);
	}
	s->pool = (const uint8_t *)pool;
	memset(&stream_config, 0, sizeof(stream_config));
	stream_config.rank = s->rank;
	stream_config.ranks = K3_STREAMED_SHARDS;
	stream_config.chunk_bytes = K3_STREAMED_STREAM_CHUNK;
	stream_config.depth = K3_STREAMED_STREAM_DEPTH;
	stream_config.fabric_gid_index = s->config.gid_index;
	stream_config.pair_gid_index = s->config.pair_gid_index;
	stream_config.traffic_class = s->config.traffic_class;
	stream_config.fabric_device = s->config.fabric_device;
	stream_config.pair_device = s->config.pair_device;
	stream_config.region = s->region;
	stream_config.region_bytes = s->region_bytes;
	stream_config.gather = K3StreamedGather;
	stream_config.gather_context = s;
	status = SparkWeightStreamOpen(&stream_config, &s->stream);
	if ( status != SPARK_STATUS_OK )
	{
		fprintf(stderr, "sparkpipe_k3: the streamed weight stream did not open status=%d (fabric %s, pair %s)\n", (int)status, s->config.fabric_device, s->config.pair_device);
		s->stream = 0;
		return status;
	}
	memset(&window_config, 0, sizeof(window_config));
	window_config.stream = s->stream;
	window_config.region = s->region;
	window_config.region_bytes = s->region_bytes;
	window_config.rank = s->rank;
	window_config.ranks = K3_STREAMED_SHARDS;
	window_config.dense_buffers = 1u;
	window_config.dense_capacity = s->window.config.dense_capacity;
	window_config.expert_capacity = s->window.config.expert_capacity;
	window_config.read_own = K3StreamedReadOwn;
	window_config.own_context = s;
	window_config.read_partner = K3StreamedReadPartner;
	window_config.partner_context = s;
	status = SparkPackWindowInitialize(&s->window, &window_config);
	if ( status != SPARK_STATUS_OK )
		return status;
	s->opened = 1u;
	return SPARK_STATUS_OK;
}

static SparkStatus K3StreamedKeys(K3Streamed *s, uint64_t keys)
{
	const uint64_t older = (uint64_t)SparkKvShardGatherKeys(s->dispatch->buffers_host->kv_shard, (uint32_t)keys) * K3GlobalKv::kSlotBytes;
	const uint64_t latent = keys * K3GlobalKv::kSlotBytes, expanded = keys * K3StreamedHeadKv<K3_STREAMED_HEADS,K3GlobalKv>::kSlotBytes;
	if ( keys <= s->key_capacity )
		return SPARK_STATUS_OK;
	cudaFree(s->mla_keys);
	s->mla_keys = 0;
	s->key_capacity = 0u;
	if ( cudaMalloc((void **)&s->mla_keys, (size_t)(latent + expanded + older + 768u)) != cudaSuccess )
	{
		s->mla_keys = 0;
		fprintf(stderr, "sparkpipe_k3: streamed prefill key buffers for %llu keys (%llu bytes) do not fit\n", (unsigned long long)keys,
			(unsigned long long)(latent + expanded + older));
		SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
	}
	s->mla.latent_keys = s->mla_keys;
	s->mla.expanded = s->mla_keys + ((latent + 255u) & ~(uint64_t)255u);
	s->mla.older_send = s->mla.expanded + ((expanded + 255u) & ~(uint64_t)255u);
	s->mla.older_receive = s->mla.expanded;
	s->mla.older_capacity = expanded;
	s->mla.key_capacity = (uint32_t)keys;
	s->key_capacity = keys;
	return SPARK_STATUS_OK;
}

static SparkStatus K3StreamedCheck(K3Streamed *s, int32_t launched, const char *what, uint32_t layer)
{
	if ( launched == LM_LAUNCH_OK && s->round.status == SPARK_STATUS_OK )
		return SPARK_STATUS_OK;
	fprintf(stderr, "sparkpipe_k3: streamed prefill %s failed at layer %u launch=%d collective=%d\n", what, layer, (int)launched, (int)s->round.status);
	return s->round.status != SPARK_STATUS_OK ? s->round.status : SPARK_STATUS_INTERNAL_ERROR;
}

static SparkStatus K3StreamedShard(K3Streamed *s, uint32_t layer, uint32_t shard, uint8_t *const *rank_base, const uint8_t *block)
{
	const K3LayerWeights *own = &s->dispatch->weights[layer];
	if ( shard == s->rank )
	{
		s->shards[shard] = *own;
		return SPARK_STATUS_OK;
	}
	if ( SparkK3ShardWeights(own, s->module->pack.payload_base, s->bound[layer], &s->ranges[layer], rank_base[shard], block, &s->shards[shard]) != SPARK_K3_DISPATCH_OK )
	{
		fprintf(stderr, "sparkpipe_k3: streamed prefill cannot place shard %u of layer %u in the window\n", shard, layer);
		SPARK_FAIL(SPARK_STATUS_PARSE_ERROR);
	}
	return SPARK_STATUS_OK;
}

static SparkStatus K3StreamedLayer(K3Streamed *s, uint32_t layer, uint32_t rows, uint32_t multiprocessors, cudaStream_t stream)
{
	const uint32_t kind = K3_LAYER_KIND(layer), kda_index = layer - layer / 4u;
	uint8_t *rank_base[K3_STREAMED_SHARDS];
	uint32_t shard, step;
	SparkStatus status;
	if ( cudaEventSynchronize(s->layer_done) != cudaSuccess )
		SPARK_FAIL(SPARK_STATUS_IO_ERROR);
	status = SparkPackWindowDense(&s->window, &s->ranges[layer], 0u, s->tag++, rank_base);
	if ( status != SPARK_STATUS_OK )
	{
		fprintf(stderr, "sparkpipe_k3: streamed prefill dense broadcast failed at layer %u status=%d\n", layer, (int)status);
		return status;
	}
	for (shard = 0u; shard < K3_STREAMED_SHARDS && status == SPARK_STATUS_OK; shard++)
		status = K3StreamedShard(s, layer, shard, rank_base, s->region + s->window.expert_slot[0]);
	if ( status != SPARK_STATUS_OK )
		return status;
	if ( kind != LM_LAYER_RECURRENT )
	{
		s->mla.store.pages = s->dispatch->mla_cache[layer / 4u];
		s->mla.store.shard = s->dispatch->buffers_host->kv_shard;
	}
	status = K3StreamedCheck(s, K3StreamedLayerAttention<LmMxfp4,K3GlobalKv,K3_STREAMED_HEADS>(&s->b, s->shards,
		kind == LM_LAYER_RECURRENT ? &s->states[(uint64_t)kda_index * K3_STREAMED_SHARDS] : 0, K3_STREAMED_SHARDS, &s->kda, kind != LM_LAYER_RECURRENT ? &s->mla : 0,
		&s->join, layer, s->partial, s->hidden_accumulator, rows, multiprocessors, stream), "attention", layer);
	if ( status != SPARK_STATUS_OK )
		return status;
	if ( layer < K3_FIRST_ROUTED_LAYER )
		status = K3StreamedCheck(s, K3StreamedDenseMlp<LmMxfp4>(&s->b, s->shards, K3_STREAMED_SHARDS, s->partial, s->hidden_accumulator, rows, multiprocessors, stream),
			"dense MLP", layer);
	else
	{
		status = K3StreamedCheck(s, K3StreamedMoeBegin<LmMxfp4>(&s->b, s->shards, K3_STREAMED_SHARDS, &s->join, s->latent_accumulator, rows, rows * K3_TOP_K,
			multiprocessors, stream), "router", layer);
		for (step = 0u; step < K3_STREAMED_SHARDS && status == SPARK_STATUS_OK; step++)
		{
			const uint8_t *block = 0;
			uint32_t source = UINT32_MAX;
			if ( step >= 2u && cudaEventSynchronize(s->slot_done[step & 1u]) != cudaSuccess )
				SPARK_FAIL(SPARK_STATUS_IO_ERROR);
			status = SparkPackWindowExpertStep(&s->window, &s->ranges[layer], step, s->tag++, &source, &block);
			if ( status != SPARK_STATUS_OK )
			{
				fprintf(stderr, "sparkpipe_k3: streamed prefill expert step %u failed at layer %u status=%d\n", step, layer, (int)status);
				return status;
			}
			if ( source >= K3_STREAMED_SHARDS || (source != s->rank && block == 0) )
				SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
			status = K3StreamedShard(s, layer, source, rank_base, block);
			if ( status == SPARK_STATUS_OK )
				status = K3StreamedCheck(s, K3StreamedMoeShard<LmMxfp4>(&s->b, &s->shards[source], source, step == 0u ? 1u : 0u, K3_STREAMED_SHARDS, &s->join,
					s->latent_accumulator, rows, rows * K3_TOP_K, multiprocessors, stream), "experts", layer);
			if ( status == SPARK_STATUS_OK && cudaEventRecord(s->slot_done[step & 1u], stream) != cudaSuccess )
				status = SPARK_STATUS_IO_ERROR;
		}
		if ( status == SPARK_STATUS_OK )
			status = K3StreamedCheck(s, K3StreamedMoeEnd<LmMxfp4>(&s->b, s->shards, K3_STREAMED_SHARDS, &s->join, s->latent_accumulator, rows, multiprocessors, stream),
				"expert combine", layer);
	}
	if ( status == SPARK_STATUS_OK && cudaEventRecord(s->layer_done, stream) != cudaSuccess )
		status = SPARK_STATUS_IO_ERROR;
	return status;
}

static SparkStatus K3StreamedStates(K3Streamed *s, uint32_t target, uint32_t to_slot, cudaStream_t stream)
{
	const SparkK3Dispatch *d = s->dispatch;
	const uint64_t heads = d->kda_rank_heads, slot_bytes = K3_KDA_RANK_STATE_SLOT_BYTES(heads, 0u);
	const uint64_t qk = heads * K3_KDA_KEY_DIM * K3_KDA_CONV_KERNEL, v = heads * K3_KDA_VALUE_DIM * K3_KDA_CONV_KERNEL;
	uint32_t layer, part;
	cudaError_t error = cudaSuccess;
	for (layer = 0u; layer < K3_KDA_LAYER_COUNT && error == cudaSuccess && s->round.status == SPARK_STATUS_OK; layer++)
	{
		const uint64_t slot = (uint64_t)layer * d->sequences + target;
		const K3StreamedKdaState *mine = &s->states[(uint64_t)layer * K3_STREAMED_SHARDS + s->rank], *first = &s->states[(uint64_t)layer * K3_STREAMED_SHARDS];
		uint8_t *const pool[4] = {d->kda_state_pool + slot * slot_bytes, (uint8_t *)(d->kda_q_window_pool + slot * qk), (uint8_t *)(d->kda_k_window_pool + slot * qk),
			(uint8_t *)(d->kda_v_window_pool + slot * v)};
		uint8_t *const kept[4] = {mine->state_pool, (uint8_t *)mine->q_window, (uint8_t *)mine->k_window, (uint8_t *)mine->v_window};
		uint8_t *const send[4] = {(uint8_t *)s->kda.summary_send, (uint8_t *)s->kda.tail_send, (uint8_t *)(s->kda.tail_send + qk), (uint8_t *)(s->kda.tail_send + 2u * qk)};
		const uint64_t bytes[4] = {slot_bytes, qk * 2u, qk * 2u, v * 2u};
		for (part = 0u; part < 4u && error == cudaSuccess; part++)
			error = cudaMemcpyAsync(to_slot != 0u ? pool[part] : send[part], to_slot != 0u ? kept[part] : pool[part], (size_t)bytes[part], cudaMemcpyDeviceToDevice, stream);
		if ( to_slot != 0u || error != cudaSuccess )
			continue;
		K3StreamedRound(&s->round, (void *)stream, SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_GATHER, 1u, (uint32_t)(slot_bytes / 2u), s->kda.summary_send, first->state_pool);
		K3StreamedRound(&s->round, (void *)stream, SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_GATHER, 1u, (uint32_t)(2u * qk + v), s->kda.tail_send, first->q_window);
	}
	if ( error != cudaSuccess )
		SPARK_FAIL(SPARK_STATUS_IO_ERROR);
	return s->round.status;
}

static uint64_t K3StreamedNowNs(void)
{
	struct timespec now;
	clock_gettime(CLOCK_MONOTONIC, &now);
	return (uint64_t)now.tv_sec * 1000000000ull + (uint64_t)now.tv_nsec;
}

static SparkStatus K3StreamedUpload(void *device, const void *host, uint64_t bytes, cudaStream_t stream)
{
	return cudaMemcpyAsync(device, host, (size_t)bytes, cudaMemcpyHostToDevice, stream) == cudaSuccess ? SPARK_STATUS_OK : SPARK_STATUS_IO_ERROR;
}

static SparkStatus K3StreamedRun(K3Streamed *s, const SparkStageRunnerStreamedPrefill *request, const SparkStageRunnerServices *services, cudaStream_t stream)
{
	const uint64_t pass_max = (uint64_t)K3_STREAMED_SHARDS * s->node_rows;
	const uint64_t keys = request != 0 ? (uint64_t)request->first_position + request->token_count : 0u;
	const uint64_t started = K3StreamedNowNs();
	uint32_t pass, passes, layer, row, rows, chunk;
	uint8_t *vocab_base[K3_STREAMED_SHARDS];
	K3StreamedVocab vocab;
	SparkStatus status;
	if ( request == 0 || request->token_ids == 0 || request->token_count == 0u || (request->token_count % (K3_STREAMED_SHARDS * K3_STREAMED_ROW_ALIGN)) != 0u ||
		request->sequence_slot >= s->dispatch->sequences || request->state_target >= s->dispatch->sequences || keys > UINT32_MAX )
	{
		fprintf(stderr, "sparkpipe_k3: streamed prefill takes a multiple of %u tokens for a known slot (got %u tokens, slot %u, state %u)\n",
			K3_STREAMED_SHARDS * K3_STREAMED_ROW_ALIGN, request != 0 ? request->token_count : 0u, request != 0 ? request->sequence_slot : 0u,
			request != 0 ? request->state_target : 0u);
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	}
	status = K3StreamedOpen(s, services);
	if ( status == SPARK_STATUS_OK )
		status = K3StreamedKeys(s, keys);
	if ( status != SPARK_STATUS_OK )
		return status;
	s->b.cache_shard.pages = s->dispatch->mla_cache[0];
	passes = (uint32_t)((request->token_count + pass_max - 1u) / pass_max);
	if ( cudaEventRecord(s->layer_done, stream) != cudaSuccess )
		SPARK_FAIL(SPARK_STATUS_IO_ERROR);
	status = K3StreamedStates(s, request->state_target, 0u, stream);
	for (pass = 0u; pass < passes && status == SPARK_STATUS_OK; pass++)
	{
		const uint64_t base = (uint64_t)request->first_position + (uint64_t)pass * pass_max;
		uint32_t *host = s->host_staging, *small;
		uint64_t node_first, pass_rows;
		rows = (uint32_t)((request->token_count - (uint64_t)pass * pass_max) / K3_STREAMED_SHARDS);
		rows = rows < s->node_rows ? rows : s->node_rows;
		pass_rows = (uint64_t)K3_STREAMED_SHARDS * rows;
		node_first = base + (uint64_t)s->rank * rows;
		if ( cudaStreamSynchronize(stream) != cudaSuccess )
			SPARK_FAIL(SPARK_STATUS_IO_ERROR);
		for (row = 0u; row < pass_rows; row++)
			host[row] = (uint32_t)(base + row);
		small = host + pass_rows;
		for (chunk = 0u; chunk <= K3_STREAMED_CHUNKS; chunk++)
			small[chunk] = chunk * (rows / K3_STREAMED_CHUNKS);
		small[K3_STREAMED_CHUNKS + 1u] = 0u;
		small[K3_STREAMED_CHUNKS + 2u] = rows;
		small[K3_STREAMED_CHUNKS + 3u] = 0u;
		small[K3_STREAMED_CHUNKS + 4u] = (uint32_t)pass_rows;
		small[K3_STREAMED_CHUNKS + 5u] = (uint32_t)(base + pass_rows);
		small[K3_STREAMED_CHUNKS + 6u] = (uint32_t)base;
		small[K3_STREAMED_CHUNKS + 7u] = request->sequence_slot;
		status = K3StreamedUpload((void *)s->mla.pass_positions, host, pass_rows * sizeof(uint32_t), stream);
		if ( status == SPARK_STATUS_OK )
			status = K3StreamedUpload(s->positions_device, host + (uint64_t)s->rank * rows, (uint64_t)rows * sizeof(uint32_t), stream);
		if ( status == SPARK_STATUS_OK )
			status = K3StreamedUpload(s->tokens_device, request->token_ids + (node_first - request->first_position), (uint64_t)rows * sizeof(uint32_t), stream);
		if ( status == SPARK_STATUS_OK )
			status = K3StreamedUpload((void *)s->kda.chunk_row_begin, small, (K3_STREAMED_CHUNKS + 1u) * sizeof(uint32_t), stream);
		if ( status == SPARK_STATUS_OK )
			status = K3StreamedUpload((void *)s->b.dense_row_offset, small + K3_STREAMED_CHUNKS + 1u, 2u * sizeof(uint32_t), stream);
		if ( status == SPARK_STATUS_OK )
			status = K3StreamedUpload((void *)s->mla.pass_row_offset, small + K3_STREAMED_CHUNKS + 3u, 2u * sizeof(uint32_t), stream);
		if ( status == SPARK_STATUS_OK )
			status = K3StreamedUpload((void *)s->mla.key_context, small + K3_STREAMED_CHUNKS + 5u, sizeof(uint32_t), stream);
		if ( status == SPARK_STATUS_OK )
			status = K3StreamedUpload(s->context_device, small + K3_STREAMED_CHUNKS + 5u, sizeof(uint32_t), stream);
		if ( status == SPARK_STATUS_OK )
			status = K3StreamedUpload(s->older_context_device + request->sequence_slot, small + K3_STREAMED_CHUNKS + 6u, sizeof(uint32_t), stream);
		if ( status == SPARK_STATUS_OK )
			status = K3StreamedUpload((void *)s->mla.older_sequence, small + K3_STREAMED_CHUNKS + 7u, sizeof(uint32_t), stream);
		for (row = 0u; row < pass_rows && status == SPARK_STATUS_OK; row++)
			host[row] = request->sequence_slot;
		if ( status == SPARK_STATUS_OK && cudaStreamSynchronize(stream) != cudaSuccess )
			status = SPARK_STATUS_IO_ERROR;
		if ( status == SPARK_STATUS_OK )
			status = K3StreamedUpload(s->store_sequence_device, host, pass_rows * sizeof(uint32_t), stream);
		if ( status != SPARK_STATUS_OK )
			break;
		s->mla.pass_rows = (uint32_t)pass_rows;
		s->mla.first_key = (uint32_t)base;
		if ( cudaEventSynchronize(s->layer_done) != cudaSuccess )
			SPARK_FAIL(SPARK_STATUS_IO_ERROR);
		status = SparkPackWindowDense(&s->window, &s->vocab, 0u, s->tag++, vocab_base);
		if ( status != SPARK_STATUS_OK )
			break;
		memset(&vocab, 0, sizeof(vocab));
		for (row = 0u; row < K3_STREAMED_SHARDS; row++)
			vocab.slice[row] = (const uint16_t *)vocab_base[row];
		vocab.slice_rows = s->vocab_rows;
		vocab.slices = K3_STREAMED_SHARDS;
		status = K3StreamedCheck(s, K3StreamedEmbed(&vocab, s->tokens_device, s->b.hidden_bf16, rows, stream), "embedding", 0u);
		if ( status == SPARK_STATUS_OK && cudaEventRecord(s->layer_done, stream) != cudaSuccess )
			status = SPARK_STATUS_IO_ERROR;
		for (layer = 0u; layer < K3_LAYERS && status == SPARK_STATUS_OK; layer++)
			status = K3StreamedLayer(s, layer, rows, request->multiprocessors, stream);
		fprintf(stderr, "sparkpipe_k3: streamed prefill pass %u/%u rows %u x %u keys to %llu status=%d at %.1f s\n", pass + 1u, passes, rows, K3_STREAMED_SHARDS,
			(unsigned long long)(base + pass_rows), (int)status, (double)(K3StreamedNowNs() - started) / 1e9);
	}
	if ( status == SPARK_STATUS_OK )
		status = cudaStreamSynchronize(stream) == cudaSuccess ? s->round.status : SPARK_STATUS_IO_ERROR;
	if ( status == SPARK_STATUS_OK )
		status = K3StreamedStates(s, request->state_target, 1u, stream);
	if ( status == SPARK_STATUS_OK && cudaStreamSynchronize(stream) != cudaSuccess )
		status = SPARK_STATUS_IO_ERROR;
	fprintf(stderr, "sparkpipe_k3: streamed prefill sequence %llu positions %u..%llu in %u passes status=%d %.1f s\n", (unsigned long long)request->sequence_id,
		request->first_position, (unsigned long long)keys, passes, (int)status, (double)(K3StreamedNowNs() - started) / 1e9);
	return status;
}
