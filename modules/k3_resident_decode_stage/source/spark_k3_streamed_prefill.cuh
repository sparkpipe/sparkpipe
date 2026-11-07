#pragma once

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "sparkpipe/spark_k3_bind.h"
#include "inference/runner/pass_prefill.cuh"
#include "inference/llms/kimi_k3/streamed.cuh"
#include "spark_k3_weight_binds.h"

#define K3_STREAMED_SHARDS LM_PASS_RANKS
#define K3_STREAMED_HEADS (K3_MLA_HEADS / K3_STREAMED_SHARDS)
#define K3_STREAMED_ROW_ALIGN 128u
#define K3_STREAMED_CHUNKS 8u
#define K3_STREAMED_KEY_BLOCK 262144u

static_assert(K3_STREAMED_ROW_ALIGN % K3_STREAMED_CHUNKS == 0u, "every node row of a pass belongs to one of the recurrent chunks");
static_assert(K3_STREAMED_SHARDS <= LM_CELL_MMA_MAX_SEGMENTS, "one expert down projection can take every shard's weights");
static_assert(K3_LAYERS <= LM_PASS_MAX_LAYERS, "the pass engine holds every layer");

struct K3Streamed
{
	LmPass pass;
	SparkK3Dispatch *dispatch;
	const SparkK3ModuleState *module;
	uint32_t moe_members;
	uint64_t key_capacity;
	const SparkK3BoundLayer *bound[K3_LAYERS];
	K3LayerWeights shards[K3_STREAMED_SHARDS];
	K3LayerBuffers b;
	K3StreamedJoin join;
	uint16_t *partial;
	float *hidden_accumulator;
	float *latent_accumulator;
	K3StreamedKdaExchange kda;
	K3StreamedKdaState states[K3_KDA_LAYER_COUNT * K3_STREAMED_SHARDS];
	K3StreamedMlaExchange mla;
	uint32_t *older_context_device;
	uint32_t *store_sequence_device;
	uint8_t *mla_keys;

	SparkStatus Begin(LmPass *p, const SparkStageRunnerStreamedPrefill *request, uint64_t padded_keys, cudaStream_t stream);
	SparkStatus PassBegin(LmPass *p, const LmPassStep *step, cudaStream_t stream);
	uint64_t HelpCapacity(void) const { return mla.help_capacity; }
	void HelpChanged(LmPass *p) { mla.help = p->balance.help_active != 0u ? p->balance.help : 0; }
	uint16_t *Hidden(void) const { return b.hidden_bf16; }
	SparkStatus Layer(LmPass *p, uint32_t layer, uint64_t dense_item, uint64_t *expert_item, uint32_t rows, uint32_t multiprocessors, cudaStream_t stream);
	void PassReport(LmPass *p, uint32_t pass);
	SparkStatus End(LmPass *p, const SparkStageRunnerStreamedPrefill *request, cudaStream_t stream);
};

static void K3StreamedDestroy(K3Streamed *s)
{
	if ( s == 0 )
		return;
	cudaFree(s->mla_keys);
	LmPassDestroy(&s->pass);
	delete s;
}

static SparkStatus K3StreamedBuffers(K3Streamed *s)
{
	LmPass *p = &s->pass;
	const SparkK3ScratchWidths *w = &s->dispatch->widths;
	const K3LayerBuffers *own = s->dispatch->buffers_host;
	const uint64_t rows = p->node_rows, pass = (uint64_t)K3_STREAMED_SHARDS * rows;
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
	b->hidden_bf16 = LmPassAlloc<uint16_t>(p, rows * K3_HIDDEN);
	b->normed_bf16 = LmPassAlloc<uint16_t>(p, rows * K3_HIDDEN);
	b->fused_qkvb_bf16 = LmPassAlloc<uint16_t>(p, rows * w->qkvb);
	b->query_bf16 = LmPassAlloc<uint16_t>(p, rows * w->query);
	b->key_bf16 = LmPassAlloc<uint16_t>(p, rows * w->key);
	b->value_bf16 = LmPassAlloc<uint16_t>(p, rows * w->value);
	b->gate_bf16 = LmPassAlloc<uint16_t>(p, rows * w->gate);
	b->decay_logit_bf16 = LmPassAlloc<uint16_t>(p, rows * w->decay);
	b->latent_bf16 = LmPassAlloc<uint16_t>(p, rows * w->latent);
	b->kv_slot_bf16 = LmPassAlloc<uint16_t>(p, rows * K3_MLA_KV_A_DIM);
	b->attention_out_bf16 = LmPassAlloc<uint16_t>(p, rows * w->attention_out);
	b->shared_out_bf16 = LmPassAlloc<uint16_t>(p, rows * K3_HIDDEN);
	b->attnres_bank_bf16 = LmPassAlloc<uint16_t>(p, (uint64_t)K3_ATTNRES_MAX_SOURCES * rows * K3_HIDDEN);
	b->attnres_partial_bf16 = LmPassAlloc<uint16_t>(p, rows * K3_HIDDEN);
	b->kda_beta_logit = LmPassAlloc<uint16_t>(p, rows * K3_KDA_HEADS);
	b->kda_write_gate_out = LmPassAlloc<float>(p, rows * K3_KDA_HEADS);
	b->gate_up_bf16 = LmPassAlloc<uint16_t>(p, rows * w->gate_up);
	b->intermediate_bf16 = LmPassAlloc<uint16_t>(p, rows * (w->intermediate > (uint64_t)K3_TOP_K * s->moe_members * own->expert_w2_input
		? w->intermediate : (uint64_t)K3_TOP_K * s->moe_members * own->expert_w2_input));
	b->kda_retention = LmPassAlloc<float>(p, rows * w->retention);
	b->router_logits = LmPassAlloc<float>(p, rows * K3_EXPERTS);
	b->attnres_score_f32 = LmPassAlloc<float>(p, rows * K3_ATTNRES_MAX_SOURCES);
	b->latent_full_bf16 = LmPassAlloc<uint16_t>(p, rows * K3_ROUTED_EXPERT_HIDDEN);
	b->shared_mid_bf16 = LmPassAlloc<uint16_t>(p, rows * w->shared_mid);
	b->route_expert = LmPassAlloc<uint32_t>(p, rows * K3_TOP_K);
	b->route_packed_row = LmPassAlloc<uint32_t>(p, rows * K3_TOP_K);
	b->route_source_token = LmPassAlloc<uint32_t>(p, rows * K3_TOP_K);
	b->route_weight = LmPassAlloc<float>(p, rows * K3_TOP_K);
	b->group_row_offset = LmPassAlloc<uint32_t>(p, K3_EXPERTS + 1u);
	b->group_tile_prefix_w1 = LmPassAlloc<uint32_t>(p, K3_EXPERTS + 1u);
	b->group_tile_prefix_w2 = LmPassAlloc<uint32_t>(p, K3_EXPERTS + 1u);
	b->dense_row_offset = LmPassAlloc<uint32_t>(p, 2u);
	b->dense_tile_prefix = LmPassAlloc<uint32_t>(p, 2u);
	b->sequence_of_row = LmPassAlloc<uint32_t>(p, rows);
	b->positions = p->positions;
	b->context_length = LmPassAlloc<uint32_t>(p, 1u);
	s->partial = LmPassAlloc<uint16_t>(p, rows * K3_HIDDEN);
	s->hidden_accumulator = LmPassAlloc<float>(p, rows * K3_HIDDEN);
	s->latent_accumulator = LmPassAlloc<float>(p, rows * K3_ROUTED_EXPERT_HIDDEN);
	s->join.weight_capacity = (uint64_t)K3_HIDDEN * (join_columns > K3_ROUTED_EXPERT_HIDDEN ? join_columns : K3_ROUTED_EXPERT_HIDDEN);
	s->join.activation_capacity = rows * join_columns;
	s->join.weight = LmPassAlloc<uint16_t>(p, s->join.weight_capacity);
	s->join.activation = LmPassAlloc<uint16_t>(p, s->join.activation_capacity);
	memset(&s->kda, 0, sizeof(s->kda));
	s->kda.node = p->rank;
	s->kda.nodes = K3_STREAMED_SHARDS;
	s->kda.chunks = K3_STREAMED_CHUNKS;
	s->kda.chunk_row_begin = LmPassAlloc<uint32_t>(p, K3_STREAMED_CHUNKS + 1u);
	s->kda.chunk_slots = LmPassAlloc<uint32_t>(p, K3_STREAMED_CHUNKS);
	s->kda.tail_send = LmPassAlloc<uint16_t>(p, tail);
	s->kda.tails = LmPassAlloc<uint16_t>(p, (uint64_t)K3_STREAMED_SHARDS * tail);
	s->kda.summary_send = LmPassAlloc<float>(p, 2u * slot_floats);
	s->kda.summaries = LmPassAlloc<float>(p, (uint64_t)K3_STREAMED_SHARDS * 2u * slot_floats);
	s->kda.ends = LmPassAlloc<float>(p, K3_STREAMED_CHUNKS * slot_floats);
	s->kda.transforms = LmPassAlloc<float>(p, K3_STREAMED_CHUNKS * slot_floats);
	s->kda.incoming = LmPassAlloc<float>(p, K3_STREAMED_CHUNKS * slot_floats);
	s->kda.running = LmPassAlloc<float>(p, 2u * slot_floats);
	s->kda.q_windows = LmPassAlloc<uint16_t>(p, K3_STREAMED_CHUNKS * heads * K3_KDA_KEY_DIM * K3_KDA_CONV_KERNEL);
	s->kda.k_windows = LmPassAlloc<uint16_t>(p, K3_STREAMED_CHUNKS * heads * K3_KDA_KEY_DIM * K3_KDA_CONV_KERNEL);
	s->kda.v_windows = LmPassAlloc<uint16_t>(p, K3_STREAMED_CHUNKS * heads * K3_KDA_VALUE_DIM * K3_KDA_CONV_KERNEL);
	s->kda.zeros = LmPassAlloc<uint16_t>(p, rows * heads * K3_KDA_VALUE_DIM);
	s->kda.scratch = LmPassAlloc<uint16_t>(p, rows * heads * K3_KDA_VALUE_DIM);
	s->kda.collective = LmPassRound;
	s->kda.context = &p->round;
	state_memory = LmPassAlloc<uint8_t>(p, (uint64_t)K3_KDA_LAYER_COUNT * K3_STREAMED_SHARDS * slot_floats * sizeof(float));
	window_memory = LmPassAlloc<uint16_t>(p, (uint64_t)K3_KDA_LAYER_COUNT * K3_STREAMED_SHARDS * (2u * heads * K3_KDA_KEY_DIM + heads * K3_KDA_VALUE_DIM) * K3_KDA_CONV_KERNEL);
	memset(&s->mla, 0, sizeof(s->mla));
	s->mla.node = p->rank;
	s->mla.nodes = K3_STREAMED_SHARDS;
	s->mla.q_latents = LmPassAlloc<uint16_t>(p, pass * K3_Q_LORA_RANK);
	s->mla.query = LmPassAlloc<uint16_t>(p, pass * K3_STREAMED_HEADS * K3_MHA_QK);
	s->mla.partials = LmPassAlloc<float>(p, pass * LmGqaShardRecordFloats(K3_STREAMED_HEADS, K3_V_HEAD_DIM, 1u));
	s->mla.running = LmPassAlloc<float>(p, pass * LmGqaShardRecordFloats(K3_STREAMED_HEADS, K3_V_HEAD_DIM, 1u));
	s->mla.merged = LmPassAlloc<uint16_t>(p, pass * K3_STREAMED_HEADS * K3_V_HEAD_DIM);
	s->mla.value_send = LmPassAlloc<uint16_t>(p, pass * K3_STREAMED_HEADS * K3_V_HEAD_DIM);
	s->mla.values = LmPassAlloc<uint16_t>(p, pass * K3_STREAMED_HEADS * K3_V_HEAD_DIM);
	s->mla.help_values = LmPassAlloc<uint16_t>(p, pass / 4u * K3_STREAMED_HEADS * K3_V_HEAD_DIM);
	s->mla.help_capacity = pass / 4u;
	s->mla.pass_positions = p->pass_positions;
	s->mla.pass_sequence_of_row = LmPassAlloc<uint32_t>(p, pass);
	s->mla.pass_row_offset = LmPassAlloc<uint32_t>(p, 2u);
	s->mla.pass_tile_prefix = LmPassAlloc<uint32_t>(p, 2u);
	s->mla.key_offset = LmPassAlloc<uint32_t>(p, s->dispatch->sequences);
	s->mla.key_context = LmPassAlloc<uint32_t>(p, 1u);
	s->mla.store_sequence_of_row = s->store_sequence_device = LmPassAlloc<uint32_t>(p, pass);
	s->mla.older_sequence = LmPassAlloc<uint32_t>(p, 1u);
	s->mla.older_offset = s->mla.key_offset;
	s->mla.older_context = s->older_context_device = LmPassAlloc<uint32_t>(p, s->dispatch->sequences);
	s->mla.collective = LmPassRound;
	s->mla.context = &p->round;
	if ( p->allocation_count == LM_PASS_MAX_ALLOCATIONS || state_memory == 0 || window_memory == 0 || s->mla.older_context == 0 || s->mla.help_values == 0 )
	{
		fprintf(stderr, "sparkpipe_k3: streamed prefill engine buffers did not fit (%u allocations, %llu bytes placed)\n", p->allocation_count,
			(unsigned long long)p->engine_bytes);
		SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
	}
	for (uint32_t index = 0u; index < p->allocation_count; index++)
		if ( p->allocations[index] == 0 )
			SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
	for (uint32_t index = 0u; index < K3_KDA_LAYER_COUNT * K3_STREAMED_SHARDS; index++)
	{
		const uint64_t qk = heads * K3_KDA_KEY_DIM * K3_KDA_CONV_KERNEL, v = heads * K3_KDA_VALUE_DIM * K3_KDA_CONV_KERNEL;
		s->states[index].state_pool = state_memory + (uint64_t)index * slot_floats * sizeof(float);
		s->states[index].q_window = window_memory + (uint64_t)index * (2u * qk + v);
		s->states[index].k_window = s->states[index].q_window + qk;
		s->states[index].v_window = s->states[index].k_window + qk;
	}
	chunk_host = p->host_staging;
	for (chunk = 0u; chunk < K3_STREAMED_CHUNKS; chunk++)
		chunk_host[chunk] = chunk;
	if ( cudaMemcpy((void *)s->kda.chunk_slots, chunk_host, K3_STREAMED_CHUNKS * sizeof(uint32_t), cudaMemcpyHostToDevice) != cudaSuccess )
		SPARK_FAIL(SPARK_STATUS_IO_ERROR);
	return SPARK_STATUS_OK;
}

static uint32_t K3StreamedExpertTensor(const char *name)
{
	const size_t length = strlen(name);
	return length >= 16u && (strcmp(name + length - 16u, "expert_w1_weight") == 0 || strcmp(name + length - 16u, "expert_w2_weight") == 0) ? 1u : 0u;
}

static SparkStatus K3StreamedCreate(const SparkStageRunnerStreamedPrefillConfig *config, SparkK3Dispatch *dispatch, const SparkK3ModuleState *module,
	const char *own_pack_path, SparkWeightdLazyPack *lazy_pack, uint32_t rank, uint64_t vocab_offset, uint64_t vocab_bytes, uint32_t vocab_rows, K3Streamed **out)
{
	K3Streamed *s;
	LmPassModelLayout layout;
	LmPassTensor *tensors;
	uint32_t tensor_first[K3_LAYERS + 1u], experts_per_layer[K3_LAYERS], balances[K3_LAYERS], layer, index, count = 0u;
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
	s = new K3Streamed;
	memset(s, 0, sizeof(*s));
	s->dispatch = dispatch;
	s->module = module;
	for (s->moe_members = K3_STREAMED_SHARDS; s->moe_members > 1u && 2u * s->moe_members > config->expert_slots; s->moe_members /= 2u)
	{
	}
	tensors = (LmPassTensor *)calloc((size_t)K3_LAYERS * SPARK_K3_BIND_MAX_NAMES + 1u, sizeof(LmPassTensor));
	if ( tensors == 0 )
	{
		delete s;
		SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
	}
	for (layer = 0u; layer < K3_LAYERS; layer++)
	{
		for (index = 0u; index < module->bound_count && s->bound[layer] == 0; index++)
			if ( module->bound[index].layer_index == layer )
				s->bound[layer] = &module->bound[index];
		tensor_first[layer] = count;
		experts_per_layer[layer] = layer >= K3_FIRST_ROUTED_LAYER ? K3_STREAMED_SHARDS : 0u;
		balances[layer] = K3_LAYER_KIND(layer) != LM_LAYER_RECURRENT ? 1u : 0u;
		if ( s->bound[layer] == 0 || s->bound[layer]->tensor_count > SPARK_K3_BIND_MAX_NAMES )
		{
			fprintf(stderr, "sparkpipe_k3: streamed prefill cannot place layer %u of the pack in window ranges\n", layer);
			free(tensors);
			delete s;
			SPARK_FAIL(SPARK_STATUS_PARSE_ERROR);
		}
		for (index = 0u; index < s->bound[layer]->tensor_count; index++)
		{
			tensors[count].start = module->pack.payload_base + s->bound[layer]->tensors[index].entry.payload_offset;
			tensors[count].bytes = s->bound[layer]->tensors[index].entry.bytes;
			tensors[count].expert = K3StreamedExpertTensor(s->bound[layer]->tensors[index].name);
			count++;
		}
	}
	tensor_first[K3_LAYERS] = count;
	memset(&layout, 0, sizeof(layout));
	layout.tag = "sparkpipe_k3";
	layout.lazy_pack = lazy_pack;
	layout.own_pack_path = own_pack_path;
	layout.pack_file_bytes = module->pack.file_bytes;
	layout.device = dispatch->device;
	layout.rank = rank;
	layout.layers = K3_LAYERS;
	layout.row_align = K3_STREAMED_ROW_ALIGN;
	layout.hidden = K3_HIDDEN;
	layout.row_elements = K3_HIDDEN;
	layout.tensors = tensors;
	layout.tensor_first = tensor_first;
	layout.experts_per_layer = experts_per_layer;
	layout.balances = balances;
	layout.vocab_offset = vocab_offset;
	layout.vocab_bytes = vocab_bytes;
	layout.vocab_rows = vocab_rows;
	status = LmPassCreate(&s->pass, config, &layout);
	free(tensors);
	if ( status != SPARK_STATUS_OK )
	{
		delete s;
		return status;
	}
	status = K3StreamedBuffers(s);
	if ( status != SPARK_STATUS_OK )
	{
		fprintf(stderr, "sparkpipe_k3: streamed prefill resources failed status=%d (own pack %s, partner pack %s)\n", (int)status,
			own_pack_path != 0 ? own_pack_path : "(none)", config->partner_pack_path);
		K3StreamedDestroy(s);
		return status;
	}
	fprintf(stderr, "sparkpipe_k3: streamed prefill node_rows=%u window=%llu (%u x dense %llu x %u ranks, %u x experts %llu) engine=%llu, expert down projections of %u shards at once, partner=%s\n",
		s->pass.node_rows, (unsigned long long)s->pass.region_bytes, s->pass.config.dense_buffers, (unsigned long long)s->pass.window.config.dense_capacity, K3_STREAMED_SHARDS,
		s->pass.config.expert_slots, (unsigned long long)s->pass.window.config.expert_capacity, (unsigned long long)s->pass.engine_bytes, s->moe_members, config->partner_pack_path);
	*out = s;
	return SPARK_STATUS_OK;
}

static SparkStatus K3StreamedKeys(K3Streamed *s, uint64_t keys)
{
	const uint64_t older = (uint64_t)SparkKvShardGatherKeys(s->dispatch->buffers_host->kv_shard, (uint32_t)keys) * K3GlobalKv::kSlotBytes;
	const uint64_t block = keys < K3_STREAMED_KEY_BLOCK ? keys : K3_STREAMED_KEY_BLOCK;
	const uint64_t latent = keys * K3GlobalKv::kSlotBytes, expanded = block * K3StreamedHeadKv<K3_STREAMED_HEADS,K3GlobalKv>::kSlotBytes;
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
	s->mla.expanded_keys = (uint32_t)block;
	s->mla.key_block = K3_STREAMED_KEY_BLOCK;
	s->mla.key_capacity = (uint32_t)keys;
	s->key_capacity = keys;
	return SPARK_STATUS_OK;
}

static SparkStatus K3StreamedShard(K3Streamed *s, uint32_t layer, uint32_t shard, uint8_t *const *rank_base, const uint8_t *block)
{
	const K3LayerWeights *own = &s->dispatch->weights[layer];
	if ( shard == s->pass.rank && s->pass.pool != 0 )
		block = s->pass.pool + s->pass.ranges[layer].experts.offset;
	if ( SparkK3ShardWeights(own, s->module->pack.payload_base, s->bound[layer], &s->pass.ranges[layer], rank_base[shard], block, &s->shards[shard]) != SPARK_K3_DISPATCH_OK )
	{
		fprintf(stderr, "sparkpipe_k3: streamed prefill cannot place shard %u of layer %u in the window\n", shard, layer);
		SPARK_FAIL(SPARK_STATUS_PARSE_ERROR);
	}
	return SPARK_STATUS_OK;
}

SparkStatus K3Streamed::Layer(LmPass *p, uint32_t layer, uint64_t dense_item, uint64_t *expert_item, uint32_t rows, uint32_t multiprocessors, cudaStream_t stream)
{
	const uint32_t kind = K3_LAYER_KIND(layer), kda_index = layer - layer / 4u;
	uint8_t *rank_base[K3_STREAMED_SHARDS];
	const void *down_weights[K3_STREAMED_SHARDS], *up_weights[K3_STREAMED_SHARDS];
	uint64_t held;
	uint32_t shard, step;
	SparkStatus status = LmPassDense(p, dense_item, rank_base);
	if ( status != SPARK_STATUS_OK )
		return status;
	for (shard = 0u; shard < K3_STREAMED_SHARDS && status == SPARK_STATUS_OK; shard++)
		status = K3StreamedShard(this, layer, shard, rank_base, p->region + p->window.expert_slot[0]);
	if ( status != SPARK_STATUS_OK )
		return status;
	mla.own_marks = p->own_marks[layer];
	if ( kind != LM_LAYER_RECURRENT )
	{
		mla.store.pages = dispatch->mla_cache[layer / 4u];
		mla.store.shard = dispatch->buffers_host->kv_shard;
	}
	if ( cudaEventRecord(p->marks[layer][0], stream) != cudaSuccess )
		SPARK_FAIL(SPARK_STATUS_IO_ERROR);
	status = LmPassCheck(p, K3StreamedLayerAttention<LmMxfp4,K3GlobalKv,K3_STREAMED_HEADS>(&b, shards,
		kind == LM_LAYER_RECURRENT ? &states[(uint64_t)kda_index * K3_STREAMED_SHARDS] : 0, K3_STREAMED_SHARDS, &kda, kind != LM_LAYER_RECURRENT ? &mla : 0,
		&join, layer, partial, hidden_accumulator, rows, multiprocessors, stream), "attention", layer);
	if ( status != SPARK_STATUS_OK || cudaEventRecord(p->marks[layer][1], stream) != cudaSuccess )
		return status != SPARK_STATUS_OK ? status : SPARK_STATUS_IO_ERROR;
	if ( layer < K3_FIRST_ROUTED_LAYER )
		status = LmPassCheck(p, K3StreamedDenseMlp<LmMxfp4>(&b, shards, K3_STREAMED_SHARDS, partial, hidden_accumulator, rows, multiprocessors, stream),
			"dense MLP", layer);
	else
	{
		status = LmPassCheck(p, K3StreamedMoeBegin<LmMxfp4>(&b, shards, K3_STREAMED_SHARDS, &join, latent_accumulator, rows, rows * K3_TOP_K,
			multiprocessors, stream), "router", layer);
		for (step = 0u; step < K3_STREAMED_SHARDS && status == SPARK_STATUS_OK; step++)
		{
			const uint64_t item = (*expert_item)++;
			const uint32_t member = step % moe_members;
			const uint8_t *block = 0;
			uint32_t source = UINT32_MAX;
			status = LmPassExpert(p, item, &source, &block);
			if ( status != SPARK_STATUS_OK )
				return status;
			status = K3StreamedShard(this, layer, source, rank_base, block);
			if ( status == SPARK_STATUS_OK )
			{
				up_weights[member] = shards[source].expert_w1_weight;
				down_weights[member] = shards[source].expert_w2_weight;
				status = LmPassCheck(p, K3StreamedMoeShard(&b, &shards[source], source, member, moe_members, K3_STREAMED_SHARDS), "experts", layer);
			}
			if ( status != SPARK_STATUS_OK || member + 1u < moe_members )
				continue;
			status = LmPassCheck(p, K3StreamedMoeUp(&b, up_weights, moe_members, rows * K3_TOP_K, stream), "expert up projection", layer);
			if ( status == SPARK_STATUS_OK )
				status = LmPassCheck(p, K3StreamedMoeDown(&b, down_weights, moe_members, step + 1u == moe_members ? 1u : 0u, latent_accumulator, rows,
					rows * K3_TOP_K, stream), "expert down projection", layer);
			for (held = item + 1u - moe_members; held <= item && status == SPARK_STATUS_OK; held++)
				status = LmPassReleaseExpert(p, held, stream);
		}
		if ( status == SPARK_STATUS_OK )
			status = LmPassCheck(p, K3StreamedMoeEnd<LmMxfp4>(&b, shards, K3_STREAMED_SHARDS, &join, latent_accumulator, rows, multiprocessors, stream),
				"expert combine", layer);
	}
	if ( status == SPARK_STATUS_OK && cudaEventRecord(p->marks[layer][2], stream) != cudaSuccess )
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
	for (layer = 0u; layer < K3_KDA_LAYER_COUNT && error == cudaSuccess && s->pass.round.status == SPARK_STATUS_OK; layer++)
	{
		const uint64_t slot = (uint64_t)layer * d->sequences + target;
		const K3StreamedKdaState *mine = &s->states[(uint64_t)layer * K3_STREAMED_SHARDS + s->pass.rank], *first = &s->states[(uint64_t)layer * K3_STREAMED_SHARDS];
		uint8_t *const pool[4] = {d->kda_state_pool + slot * slot_bytes, (uint8_t *)(d->kda_q_window_pool + slot * qk), (uint8_t *)(d->kda_k_window_pool + slot * qk),
			(uint8_t *)(d->kda_v_window_pool + slot * v)};
		uint8_t *const kept[4] = {mine->state_pool, (uint8_t *)mine->q_window, (uint8_t *)mine->k_window, (uint8_t *)mine->v_window};
		uint8_t *const send[4] = {(uint8_t *)s->kda.summary_send, (uint8_t *)s->kda.tail_send, (uint8_t *)(s->kda.tail_send + qk), (uint8_t *)(s->kda.tail_send + 2u * qk)};
		const uint64_t bytes[4] = {slot_bytes, qk * 2u, qk * 2u, v * 2u};
		for (part = 0u; part < 4u && error == cudaSuccess; part++)
			error = cudaMemcpyAsync(to_slot != 0u ? pool[part] : send[part], to_slot != 0u ? kept[part] : pool[part], (size_t)bytes[part], cudaMemcpyDeviceToDevice, stream);
		if ( to_slot != 0u || error != cudaSuccess )
			continue;
		LmPassRound(&s->pass.round, (void *)stream, SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_GATHER, 1u, (uint32_t)(slot_bytes / 2u), s->kda.summary_send, first->state_pool);
		LmPassRound(&s->pass.round, (void *)stream, SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_GATHER, 1u, (uint32_t)(2u * qk + v), s->kda.tail_send, first->q_window);
	}
	if ( error != cudaSuccess )
		SPARK_FAIL(SPARK_STATUS_IO_ERROR);
	return s->pass.round.status;
}

SparkStatus K3Streamed::Begin(LmPass *p, const SparkStageRunnerStreamedPrefill *request, uint64_t padded_keys, cudaStream_t stream)
{
	SparkStatus status;
	(void)p;
	if ( request->sequence_slot >= dispatch->sequences || request->state_target >= dispatch->sequences )
	{
		fprintf(stderr, "sparkpipe_k3: streamed prefill takes at least one token for a known slot (got %u tokens, slot %u, state %u)\n",
			request->token_count, request->sequence_slot, request->state_target);
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	}
	status = K3StreamedKeys(this, padded_keys);
	if ( status != SPARK_STATUS_OK )
		return status;
	b.cache_shard.pages = dispatch->mla_cache[0];
	b.route_fault = request->route_fault;
	return K3StreamedStates(this, request->state_target, 0u, stream);
}

SparkStatus K3Streamed::PassBegin(LmPass *p, const LmPassStep *step, cudaStream_t stream)
{
	uint32_t *small = p->host_staging, chunk, row;
	SparkStatus status;
	for (chunk = 0u; chunk <= K3_STREAMED_CHUNKS; chunk++)
		small[chunk] = chunk * (step->rows / K3_STREAMED_CHUNKS);
	small[K3_STREAMED_CHUNKS + 1u] = 0u;
	small[K3_STREAMED_CHUNKS + 2u] = step->rows;
	small[K3_STREAMED_CHUNKS + 3u] = 0u;
	small[K3_STREAMED_CHUNKS + 4u] = step->pass_rows;
	small[K3_STREAMED_CHUNKS + 5u] = (uint32_t)(step->base + step->pass_tokens);
	small[K3_STREAMED_CHUNKS + 6u] = (uint32_t)step->base;
	small[K3_STREAMED_CHUNKS + 7u] = step->sequence_slot;
	status = LmPassUpload((void *)kda.chunk_row_begin, small, (K3_STREAMED_CHUNKS + 1u) * sizeof(uint32_t), stream);
	if ( status == SPARK_STATUS_OK )
		status = LmPassUpload((void *)b.dense_row_offset, small + K3_STREAMED_CHUNKS + 1u, 2u * sizeof(uint32_t), stream);
	if ( status == SPARK_STATUS_OK )
		status = LmPassUpload((void *)mla.pass_row_offset, small + K3_STREAMED_CHUNKS + 3u, 2u * sizeof(uint32_t), stream);
	if ( status == SPARK_STATUS_OK )
		status = LmPassUpload((void *)mla.key_context, small + K3_STREAMED_CHUNKS + 5u, sizeof(uint32_t), stream);
	if ( status == SPARK_STATUS_OK )
		status = LmPassUpload((void *)b.context_length, small + K3_STREAMED_CHUNKS + 5u, sizeof(uint32_t), stream);
	if ( status == SPARK_STATUS_OK )
		status = LmPassUpload(older_context_device + step->sequence_slot, small + K3_STREAMED_CHUNKS + 6u, sizeof(uint32_t), stream);
	if ( status == SPARK_STATUS_OK )
		status = LmPassUpload((void *)mla.older_sequence, small + K3_STREAMED_CHUNKS + 7u, sizeof(uint32_t), stream);
	if ( status == SPARK_STATUS_OK && cudaStreamSynchronize(stream) != cudaSuccess )
		status = SPARK_STATUS_IO_ERROR;
	for (row = 0u; row < step->pass_rows && status == SPARK_STATUS_OK; row++)
		small[row] = step->sequence_slot;
	if ( status == SPARK_STATUS_OK )
		status = LmPassUpload(store_sequence_device, small, (uint64_t)step->pass_rows * sizeof(uint32_t), stream);
	if ( status == SPARK_STATUS_OK && cudaStreamSynchronize(stream) != cudaSuccess )
		status = SPARK_STATUS_IO_ERROR;
	if ( status != SPARK_STATUS_OK )
		return status;
	mla.pass_rows = step->pass_rows;
	mla.pass_tokens = (uint32_t)step->pass_tokens;
	kda.pass_tokens = (uint32_t)step->pass_tokens;
	mla.first_key = (uint32_t)step->base;
	return SPARK_STATUS_OK;
}

void K3Streamed::PassReport(LmPass *p, uint32_t pass)
{
	float gpu[4] = {0.0f, 0.0f, 0.0f, 0.0f}, part;
	uint32_t layer;
	for (layer = 0u; layer < K3_LAYERS; layer++)
	{
		if ( cudaEventElapsedTime(&part, p->marks[layer][0], p->marks[layer][1]) == cudaSuccess )
			gpu[K3_LAYER_KIND(layer) == LM_LAYER_RECURRENT ? 1u : 0u] += part;
		if ( cudaEventElapsedTime(&part, p->marks[layer][1], p->marks[layer][2]) == cudaSuccess )
			gpu[2] += part;
		if ( layer + 1u < K3_LAYERS && cudaEventElapsedTime(&part, p->marks[layer][2], p->marks[layer + 1u][0]) == cudaSuccess )
			gpu[3] += part;
	}
	fprintf(stderr, "sparkpipe_k3: streamed prefill pass %u GPU: MLA attention %.1f s, KDA attention %.1f s, MLP and experts %.1f s, between layers %.1f s\n", pass + 1u,
		gpu[0] / 1e3, gpu[1] / 1e3, gpu[2] / 1e3, gpu[3] / 1e3);
}

SparkStatus K3Streamed::End(LmPass *p, const SparkStageRunnerStreamedPrefill *request, cudaStream_t stream)
{
	(void)p;
	return K3StreamedStates(this, request->state_target, 1u, stream);
}

static SparkStatus K3StreamedRun(K3Streamed *s, const SparkStageRunnerStreamedPrefill *request, const SparkStageRunnerServices *services, cudaStream_t stream)
{
	return LmPassRun(&s->pass, s, request, services, stream);
}
