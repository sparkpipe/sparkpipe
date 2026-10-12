#pragma once

typedef struct Glm5NextBuffer
{
	void **pointer;
	uint64_t bytes;
} Glm5NextBuffer;

typedef struct Glm5NextModelState
{
	const SparkStageRunnerServices *services;
	uint32_t tp_degree;
	uint32_t tp_rank;
	uint32_t max_rows;
	uint32_t slots;
	uint32_t context_capacity;
	uint32_t collective_capacity;
	uint64_t file_bytes;
	uint32_t entry_count;
	Glm5NextPackEntry *entries;
	Glm5NextLayerWeights layers[G5N_LAYERS];
	uint64_t global_seen;
	const void *final_norm;
	uint64_t final_norm_offset;
	uint64_t expert_w1_offset[G5N_LAYERS];
	uint64_t expert_w2_offset[G5N_LAYERS];
	uint64_t kda_state_slot_bytes;
	uint64_t kda_window_slot_bytes;
	uint8_t *kda_state;
	uint8_t *kda_windows;
	LmKvAccessError *kv_error;
	LmKvShardView latent_views[G5N_DSA_LAYERS];
	LmKvShardView index_views[G5N_DSA_LAYERS];
	uint32_t kv_attached;
	uint32_t *dense_row_offset;
	uint32_t *dense_tile_prefix;
	uint16_t *streams;
	uint16_t *snapshot;
	uint16_t *normed;
	uint16_t *collapsed;
	uint16_t *streams_mean;
	float *hc_mixes;
	float *hc_pre;
	float *hc_post;
	float *hc_comb;
	uint16_t *q_compressed;
	uint16_t *q;
	uint16_t *query_latent;
	uint16_t *index_query;
	uint16_t *index_key;
	uint16_t *index_head_weight;
	uint16_t *index_gate;
	uint16_t *index_packed;
	uint32_t *selected_pools;
	uint32_t *selected_positions;
	float *selection_scores;
	float *index_local_scores;
	float *index_gathered_scores;
	uint16_t *fused_qkvb;
	uint16_t *fused_decay_gate;
	uint16_t *kda_decay_latent;
	uint16_t *kda_gate_latent;
	uint16_t *kda_beta_logit;
	uint16_t *kda_gate;
	uint16_t *kda_decay_logit;
	float *kda_retention;
	float *kda_write_gate;
	uint16_t *kv_slot;
	uint16_t *attention_latent;
	uint16_t *attention_value;
	uint16_t *attention_out;
	uint16_t *gate_up;
	uint16_t *intermediate;
	uint16_t *expert_out;
	uint16_t *shared_out;
	float *router_logits;
	uint32_t *route_expert;
	float *route_weight;
	uint32_t *route_source_token;
	uint32_t *route_packed_row;
	uint32_t *group_row_offset;
	uint32_t *group_tile_prefix_w1;
	uint32_t *group_tile_prefix_w2;
	uint16_t *shard_query_gathered;
	float *shard_partials;
	float *shard_partials_received;
	uint64_t steps;
	uint64_t prefill_rows;
	struct Glm5NextModelState *lane_scratch;
	struct Glm5NextModelState *lane_states[SPARK_STAGE_RUNNER_LANES];
	uint64_t lane_base[SPARK_STAGE_RUNNER_LANES][2];
	uint32_t lane_bound[SPARK_STAGE_RUNNER_LANES];
} Glm5NextModelState;

static uint32_t Glm5NextRankHeads(uint32_t total, uint32_t degree)
{
	return total / degree;
}

static uint32_t Glm5NextLocalStride(const Glm5NextModelState *state, uint32_t context)
{
	return SparkGlm5NextIndexCpLocalStride(SparkGlm5NextIndexCpPools(context), state->tp_degree);
}

static uint32_t Glm5NextKdaOrdinal(uint32_t layer)
{
	uint32_t ordinal = 0u;
	for ( uint32_t index = 0u; index < layer; ++index )
		ordinal += SPARK_GLM5_NEXT_MODEL_LAYER_IS_KDA(index) ? 1u : 0u;
	return SPARK_GLM5_NEXT_MODEL_LAYER_IS_KDA(layer) ? ordinal : G5N_NO_ORDINAL;
}

static uint32_t Glm5NextDsaOrdinal(uint32_t layer)
{
	return SPARK_GLM5_NEXT_MODEL_LAYER_IS_KDA(layer) ? G5N_NO_ORDINAL : layer / SPARK_GLM5_NEXT_MODEL_ATTENTION_PERIOD;
}

static uint64_t Glm5NextExpectedMask(uint32_t layer)
{
	uint64_t mask = 0u;
	for ( uint32_t kind = GLM5_NEXT_T_ATTN_NORM; kind < GLM5_NEXT_T_COUNT; ++kind )
		if ( Glm5NextPackKindAllowed(kind, layer) != 0u )
			mask |= UINT64_C(1) << kind;
	return mask;
}

static SparkStatus Glm5NextPackRead(FILE *file, uint64_t offset, void *destination, uint64_t bytes)
{
	if ( fseeko(file, (off_t)offset, SEEK_SET) != 0 || fread(destination, 1u, (size_t)bytes, file) != (size_t)bytes )
		SPARK_FAIL(SPARK_STATUS_IO_ERROR);
	return SPARK_STATUS_OK;
}

static SparkStatus Glm5NextPackHeaderCheck(const Glm5NextModelState *state, const Glm5NextPackHeader *header, uint32_t expert_codec)
{
	if ( header->magic != GLM5_NEXT_PACK_MAGIC || header->format_version != GLM5_NEXT_PACK_VERSION ||
		header->header_bytes != sizeof(Glm5NextPackHeader) || header->directory_entry_bytes != sizeof(Glm5NextPackEntry) ||
		header->codec_abi_version != SPARK_WEIGHT_CODEC_ABI_VERSION )
	{
		fprintf(stderr, "sparkpipe_glm5_next: the rank pack is not a GLM stage pack v1 (magic 0x%08x version %u)\n", header->magic,
			header->format_version);
		SPARK_FAIL(SPARK_STATUS_ABI_MISMATCH);
	}
	if ( (header->flags & ~GLM5_NEXT_PACK_FLAG_MTP) != 0u || header->tp_degree != state->tp_degree || header->tp_rank != state->tp_rank ||
		header->stage_count != 1u || header->stage_index != 0u || header->first_layer_index != 0u || header->layer_count != G5N_LAYERS ||
		header->total_layer_count != G5N_LAYERS || header->hidden_dimension != G5N_HIDDEN || header->vocab_count != G5N_VOCAB ||
		header->routed_expert_count != G5N_EXPERTS || header->tensor_count == 0u || header->tensor_count > GLM5_NEXT_PACK_MAX_TENSORS )
	{
		fprintf(stderr, "sparkpipe_glm5_next: the pack covers tp %u/%u stage %u/%u layers %u+%u of %u, the runner needs tp %u/%u and all %u layers\n",
			header->tp_rank, header->tp_degree, header->stage_index, header->stage_count, header->first_layer_index, header->layer_count,
			header->total_layer_count, state->tp_rank, state->tp_degree, G5N_LAYERS);
		SPARK_FAIL(SPARK_STATUS_SCHEMA_ERROR);
	}
	if ( header->linear_weight_codec != SPARK_WEIGHT_CODEC_BF16 || header->expert_weight_codec != expert_codec ||
		header->kv_cache_codec != SPARK_WEIGHT_CODEC_BF16 )
	{
		fprintf(stderr, "sparkpipe_glm5_next: pack codecs linear=%u expert=%u kv=%u, this driver is built for bf16/%u/bf16\n",
			header->linear_weight_codec, header->expert_weight_codec, header->kv_cache_codec, expert_codec);
		SPARK_FAIL(SPARK_STATUS_TARGET_MISMATCH);
	}
	if ( header->file_bytes != state->file_bytes || header->directory_offset < header->header_bytes ||
		header->directory_offset % GLM5_NEXT_PACK_ALIGNMENT != 0u ||
		header->directory_offset + (uint64_t)header->tensor_count * sizeof(Glm5NextPackEntry) > state->file_bytes )
		SPARK_FAIL(SPARK_STATUS_SCHEMA_ERROR);
	return SPARK_STATUS_OK;
}

static SparkStatus Glm5NextPackAssign(Glm5NextModelState *state, const Glm5NextPackEntry *entry)
{
	Glm5NextLayerWeights *w;
	uint64_t *seen;
	const uint64_t bit = UINT64_C(1) << entry->tensor_kind;
	if ( entry->layer_index == GLM5_NEXT_PACK_GLOBAL_LAYER )
	{
		if ( (state->global_seen & bit) != 0u )
			SPARK_FAIL(SPARK_STATUS_DUPLICATE);
		state->global_seen |= bit;
		if ( entry->tensor_kind == GLM5_NEXT_T_FINAL_NORM )
			state->final_norm_offset = entry->payload_offset;
		return SPARK_STATUS_OK;
	}
	if ( entry->layer_index >= G5N_LAYERS )
		return SPARK_STATUS_OK;
	w = &state->layers[entry->layer_index];
	seen = &w->seen;
	if ( (*seen & bit) != 0u )
		SPARK_FAIL(SPARK_STATUS_DUPLICATE);
	*seen |= bit;
	if ( entry->tensor_kind == GLM5_NEXT_T_EXPERT_UP_GATE )
	{
		w->expert_w1_offset = entry->payload_offset;
		w->expert_w1_scale_offset = entry->scale_offset;
		state->expert_w1_offset[entry->layer_index] = entry->payload_offset;
	}
	if ( entry->tensor_kind == GLM5_NEXT_T_EXPERT_DOWN )
	{
		w->expert_w2_offset = entry->payload_offset;
		w->expert_w2_scale_offset = entry->scale_offset;
		state->expert_w2_offset[entry->layer_index] = entry->payload_offset;
	}
	return SPARK_STATUS_OK;
}

static SparkStatus Glm5NextPackOpen(Glm5NextModelState *state, const char *path, uint32_t expert_codec, SparkStageRunnerModelGeometry *geometry)
{
	Glm5NextPackHeader header;
	FILE *file = fopen(path, "rb");
	SparkStatus status;
	uint32_t index, layer, mtp = 0u;
	uint64_t mtp_seen = 0u;
	if ( file == 0 )
	{
		fprintf(stderr, "sparkpipe_glm5_next: rank pack %s cannot be opened\n", path);
		SPARK_FAIL(SPARK_STATUS_NOT_FOUND);
	}
	memset(&header, 0, sizeof(header));
	status = fseeko(file, 0, SEEK_END) == 0 ? SPARK_STATUS_OK : SPARK_STATUS_IO_ERROR;
	if ( status == SPARK_STATUS_OK )
		state->file_bytes = (uint64_t)ftello(file);
	if ( status == SPARK_STATUS_OK )
		status = Glm5NextPackRead(file, 0u, &header, sizeof(header));
	if ( status == SPARK_STATUS_OK )
		status = Glm5NextPackHeaderCheck(state, &header, expert_codec);
	if ( status == SPARK_STATUS_OK )
	{
		state->entry_count = header.tensor_count;
		state->entries = (Glm5NextPackEntry *)calloc(header.tensor_count, sizeof(Glm5NextPackEntry));
		status = state->entries != 0 ? Glm5NextPackRead(file, header.directory_offset, state->entries,
			(uint64_t)header.tensor_count * sizeof(Glm5NextPackEntry)) : SPARK_STATUS_CAPACITY_EXCEEDED;
	}
	(void)fclose(file);
	for ( index = 0u; status == SPARK_STATUS_OK && index < state->entry_count; ++index )
	{
		const Glm5NextPackEntry *entry = &state->entries[index];
		const int32_t check = Glm5NextPackEntryCheck(entry, expert_codec, state->tp_degree, state->file_bytes);
		if ( check != 0 )
		{
			fprintf(stderr, "sparkpipe_glm5_next: pack entry %u (kind %u layer %u) fails its shape check (%d)\n", index, entry->tensor_kind,
				entry->layer_index, check);
			status = SPARK_STATUS_SCHEMA_ERROR;
			break;
		}
		if ( entry->layer_index == SPARK_GLM5_NEXT_MODEL_MTP_LAYER_INDEX )
		{
			if ( (mtp_seen & (UINT64_C(1) << entry->tensor_kind)) != 0u )
				status = SPARK_STATUS_DUPLICATE;
			mtp_seen |= UINT64_C(1) << entry->tensor_kind;
			mtp = 1u;
		}
		if ( status == SPARK_STATUS_OK )
			status = Glm5NextPackAssign(state, entry);
	}
	if ( status != SPARK_STATUS_OK )
		return status;
	if ( state->global_seen != ((UINT64_C(1) << GLM5_NEXT_T_EMBEDDING) | (UINT64_C(1) << GLM5_NEXT_T_FINAL_NORM) | (UINT64_C(1) << GLM5_NEXT_T_LM_HEAD)) ||
		mtp != ((header.flags & GLM5_NEXT_PACK_FLAG_MTP) != 0u ? 1u : 0u) ||
		(mtp != 0u && mtp_seen != Glm5NextExpectedMask(SPARK_GLM5_NEXT_MODEL_MTP_LAYER_INDEX)) )
	{
		fprintf(stderr, "sparkpipe_glm5_next: the pack's global or MTP tensor inventory is incomplete\n");
		SPARK_FAIL(SPARK_STATUS_SCHEMA_ERROR);
	}
	for ( layer = 0u; layer < G5N_LAYERS; ++layer )
		if ( state->layers[layer].seen != Glm5NextExpectedMask(layer) )
		{
			fprintf(stderr, "sparkpipe_glm5_next: layer %u holds tensor kinds 0x%llx, the model needs 0x%llx\n", layer,
				(unsigned long long)state->layers[layer].seen, (unsigned long long)Glm5NextExpectedMask(layer));
			SPARK_FAIL(SPARK_STATUS_SCHEMA_ERROR);
		}
	for ( index = 0u; index < state->entry_count; ++index )
	{
		const Glm5NextPackEntry *entry = &state->entries[index];
		if ( entry->tensor_kind == GLM5_NEXT_T_EMBEDDING )
		{
			geometry->embed_offset = entry->payload_offset;
			geometry->embed_bytes = entry->payload_bytes;
			geometry->embed_rows = entry->rows;
		}
		else if ( entry->tensor_kind == GLM5_NEXT_T_LM_HEAD )
		{
			geometry->head_offset = entry->payload_offset;
			geometry->head_bytes = entry->payload_bytes;
			geometry->head_rows = entry->rows;
		}
	}
	return SPARK_STATUS_OK;
}

static SparkStatus Glm5NextAllocate(Glm5NextModelState *state, uint32_t lane)
{
	const uint64_t R = state->max_rows, P = R * G5N_TOP_K, D = state->tp_degree;
	const uint32_t local_stride = Glm5NextLocalStride(state, state->context_capacity);
	const uint64_t gather_floats = (uint64_t)SparkGlm5NextIndexCpGatherSequences(state->max_rows, local_stride) * SPARK_GLM5_NEXT_INDEX_CP_SEQUENCE_FLOATS;
	const uint64_t query_stride = SparkGlm5NextKvShardQueryStride(state->max_rows, state->tp_degree);
	const uint64_t partial_stride = SparkGlm5NextKvShardPartialStrideCapacity(state->tp_degree, state->max_rows);
	const uint64_t kda_heads = Glm5NextRankHeads(G5N_KDA_HEADS, state->tp_degree), rank_qk = kda_heads * G5N_KDA_KEY;
	const uint64_t mla_heads = Glm5NextRankHeads(G5N_HEADS, state->tp_degree);
	const uint64_t q_width = mla_heads * G5N_NOPE > rank_qk ? mla_heads * G5N_NOPE : rank_qk;
	const uint64_t query_latent = R * mla_heads * G5N_LATENT > query_stride ? R * mla_heads * G5N_LATENT : query_stride;
	const uint64_t gate_up = P * (2u * G5N_EXPERT_INTER / D) > R * (2u * G5N_DENSE_INTER / D) ? P * (2u * G5N_EXPERT_INTER / D) : R * (2u * G5N_DENSE_INTER / D);
	const uint64_t inter = P * (G5N_EXPERT_INTER / D) > R * (G5N_DENSE_INTER / D) ? P * (G5N_EXPERT_INTER / D) : R * (G5N_DENSE_INTER / D);
	const Glm5NextBuffer plan[] =
	{
		{ (void **)&state->dense_row_offset, 8u }, { (void **)&state->dense_tile_prefix, 8u },
		{ (void **)&state->kv_error, lane == 0u ? sizeof(LmKvAccessError) : 0u },
		{ (void **)&state->streams, R * G5N_HC_FLAT * 2u }, { (void **)&state->snapshot, R * G5N_HC_FLAT * 2u },
		{ (void **)&state->normed, R * G5N_HIDDEN * 2u }, { (void **)&state->collapsed, R * G5N_HIDDEN * 2u },
		{ (void **)&state->streams_mean, R * G5N_HIDDEN * 2u },
		{ (void **)&state->hc_mixes, R * G5N_HC_MIX * 4u }, { (void **)&state->hc_pre, R * G5N_HC * 4u },
		{ (void **)&state->hc_post, R * G5N_HC * 4u }, { (void **)&state->hc_comb, R * G5N_HC * G5N_HC * 4u },
		{ (void **)&state->q_compressed, R * G5N_QA * 2u }, { (void **)&state->q, R * q_width * 2u },
		{ (void **)&state->query_latent, query_latent * 2u },
		{ (void **)&state->index_query, R * G5N_INDEX_QUERY * 2u }, { (void **)&state->index_key, R * G5N_INDEX_DIM * 2u },
		{ (void **)&state->index_head_weight, R * G5N_INDEX_HEADS * 2u }, { (void **)&state->index_gate, R * G5N_INDEX_DIM * 2u },
		{ (void **)&state->index_packed, R * G5N_INDEX_PACKED * 2u },
		{ (void **)&state->selected_pools, R * (G5N_SELECTED / G5N_KPOOL) * 4u }, { (void **)&state->selected_positions, R * G5N_INDEX_WIDTH * 4u },
		{ (void **)&state->selection_scores, R * SparkGlm5NextIndexCpPools(state->context_capacity) * 4u },
		{ (void **)&state->index_local_scores, gather_floats * 4u }, { (void **)&state->index_gathered_scores, D * gather_floats * 4u },
		{ (void **)&state->fused_qkvb, R * (3u * rank_qk + kda_heads) * 2u }, { (void **)&state->fused_decay_gate, R * 2u * G5N_KDA_LOW_RANK * 2u },
		{ (void **)&state->kda_decay_latent, R * G5N_KDA_LOW_RANK * 2u }, { (void **)&state->kda_gate_latent, R * G5N_KDA_LOW_RANK * 2u },
		{ (void **)&state->kda_beta_logit, R * kda_heads * 2u }, { (void **)&state->kda_gate, R * rank_qk * 2u },
		{ (void **)&state->kda_decay_logit, R * rank_qk * 2u }, { (void **)&state->kda_retention, R * rank_qk * 4u },
		{ (void **)&state->kda_write_gate, R * kda_heads * 4u },
		{ (void **)&state->kv_slot, R * (rank_qk > G5N_LATENT ? rank_qk : G5N_LATENT) * 2u },
		{ (void **)&state->attention_latent, R * mla_heads * G5N_LATENT * 2u }, { (void **)&state->attention_value, R * mla_heads * G5N_VALUE * 2u },
		{ (void **)&state->attention_out, R * G5N_HIDDEN * 2u },
		{ (void **)&state->gate_up, gate_up * 2u }, { (void **)&state->intermediate, inter * 2u },
		{ (void **)&state->expert_out, P * G5N_HIDDEN * 2u }, { (void **)&state->shared_out, R * G5N_HIDDEN * 2u },
		{ (void **)&state->router_logits, R * G5N_EXPERTS * 4u },
		{ (void **)&state->route_expert, P * 4u }, { (void **)&state->route_weight, P * 4u },
		{ (void **)&state->route_source_token, P * 4u }, { (void **)&state->route_packed_row, P * 4u },
		{ (void **)&state->group_row_offset, (G5N_EXPERTS + 1u) * 4u }, { (void **)&state->group_tile_prefix_w1, (G5N_EXPERTS + 1u) * 4u },
		{ (void **)&state->group_tile_prefix_w2, (G5N_EXPERTS + 1u) * 4u },
		{ (void **)&state->shard_query_gathered, D * query_stride * 2u },
		{ (void **)&state->shard_partials, D * partial_stride * 4u }, { (void **)&state->shard_partials_received, D * partial_stride * 4u },
		{ (void **)&state->kda_state, lane == 0u ? (uint64_t)G5N_KDA_LAYERS * state->slots * state->kda_state_slot_bytes : 0u },
		{ (void **)&state->kda_windows, lane == 0u ? 3u * (uint64_t)G5N_KDA_LAYERS * state->slots * state->kda_window_slot_bytes : 0u },
	};
	for ( uint32_t index = 0u; index < sizeof(plan) / sizeof(plan[0]); ++index )
		if ( plan[index].bytes == 0u )
			*plan[index].pointer = 0;
		else if ( cudaMalloc(plan[index].pointer, (size_t)plan[index].bytes) != cudaSuccess ||
			cudaMemset(*plan[index].pointer, 0, (size_t)plan[index].bytes) != cudaSuccess )
		{
			*plan[index].pointer = 0;
			fprintf(stderr, "sparkpipe_glm5_next: scratch buffer %u of %llu bytes does not fit\n", index, (unsigned long long)plan[index].bytes);
			SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
		}
	return SPARK_STATUS_OK;
}

static uint32_t Glm5NextScratch(Glm5NextModelState *state, void ***fields)
{
	void **pointers[] =
	{
		(void **)&state->dense_row_offset, (void **)&state->dense_tile_prefix, (void **)&state->streams, (void **)&state->snapshot,
		(void **)&state->normed, (void **)&state->collapsed, (void **)&state->streams_mean, (void **)&state->hc_mixes, (void **)&state->hc_pre,
		(void **)&state->hc_post, (void **)&state->hc_comb, (void **)&state->q_compressed, (void **)&state->q, (void **)&state->query_latent,
		(void **)&state->index_query, (void **)&state->index_key, (void **)&state->index_head_weight, (void **)&state->index_gate,
		(void **)&state->index_packed, (void **)&state->selected_pools, (void **)&state->selected_positions, (void **)&state->selection_scores,
		(void **)&state->index_local_scores, (void **)&state->index_gathered_scores, (void **)&state->fused_qkvb, (void **)&state->fused_decay_gate,
		(void **)&state->kda_decay_latent, (void **)&state->kda_gate_latent, (void **)&state->kda_beta_logit, (void **)&state->kda_gate,
		(void **)&state->kda_decay_logit, (void **)&state->kda_retention, (void **)&state->kda_write_gate, (void **)&state->kv_slot,
		(void **)&state->attention_latent, (void **)&state->attention_value, (void **)&state->attention_out, (void **)&state->gate_up,
		(void **)&state->intermediate, (void **)&state->expert_out, (void **)&state->shared_out, (void **)&state->router_logits,
		(void **)&state->route_expert, (void **)&state->route_weight, (void **)&state->route_source_token, (void **)&state->route_packed_row,
		(void **)&state->group_row_offset, (void **)&state->group_tile_prefix_w1, (void **)&state->group_tile_prefix_w2,
		(void **)&state->shard_query_gathered, (void **)&state->shard_partials, (void **)&state->shard_partials_received,
	};
	const uint32_t count = (uint32_t)(sizeof(pointers) / sizeof(pointers[0]));
	for ( uint32_t index = 0u; index < count; ++index )
		fields[index] = pointers[index];
	return count;
}

#define G5N_SCRATCH_FIELDS 64u

static void Glm5NextLaneFree(Glm5NextModelState *state)
{
	void **fields[G5N_SCRATCH_FIELDS];
	Glm5NextModelState *donor = state->lane_scratch;
	if ( donor != 0 )
	{
		const uint32_t count = Glm5NextScratch(donor, fields);
		for ( uint32_t index = 0u; index < count; ++index )
			if ( *fields[index] != 0 )
				(void)cudaFree(*fields[index]);
		free(donor);
	}
	for ( uint32_t lane = 0u; lane < SPARK_STAGE_RUNNER_LANES; ++lane )
		free(state->lane_states[lane]);
	state->lane_scratch = 0;
}

static SparkStatus Glm5NextLaneAllocate(Glm5NextModelState *state)
{
	void **fields[G5N_SCRATCH_FIELDS];
	Glm5NextModelState *donor = (Glm5NextModelState *)malloc(sizeof(*donor));
	if ( donor == 0 )
		SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
	memcpy(donor, state, sizeof(*donor));
	for ( uint32_t index = 0u, count = Glm5NextScratch(donor, fields); index < count; ++index )
		*fields[index] = 0;
	state->lane_scratch = donor;
	for ( uint32_t lane = 0u; lane < SPARK_STAGE_RUNNER_LANES; ++lane )
		if ( (state->lane_states[lane] = (Glm5NextModelState *)calloc(1u, sizeof(Glm5NextModelState))) == 0 )
			SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
	return Glm5NextAllocate(donor, 1u);
}

static void Glm5NextFree(Glm5NextModelState *state)
{
	Glm5NextLaneFree(state);
	void *pointers[] =
	{
		state->dense_row_offset, state->dense_tile_prefix, state->kv_error, state->streams, state->snapshot, state->normed, state->collapsed,
		state->streams_mean, state->hc_mixes, state->hc_pre, state->hc_post, state->hc_comb, state->q_compressed, state->q, state->query_latent,
		state->index_query, state->index_key, state->index_head_weight, state->index_gate, state->index_packed, state->selected_pools,
		state->selected_positions, state->selection_scores, state->index_local_scores, state->index_gathered_scores, state->fused_qkvb,
		state->fused_decay_gate, state->kda_decay_latent, state->kda_gate_latent, state->kda_beta_logit, state->kda_gate, state->kda_decay_logit,
		state->kda_retention, state->kda_write_gate, state->kv_slot, state->attention_latent, state->attention_value, state->attention_out,
		state->gate_up, state->intermediate, state->expert_out, state->shared_out, state->router_logits, state->route_expert, state->route_weight,
		state->route_source_token, state->route_packed_row, state->group_row_offset, state->group_tile_prefix_w1, state->group_tile_prefix_w2,
		state->shard_query_gathered, state->shard_partials, state->shard_partials_received, state->kda_state, state->kda_windows,
	};
	for ( uint32_t index = 0u; index < sizeof(pointers) / sizeof(pointers[0]); ++index )
		if ( pointers[index] != 0 )
			(void)cudaFree(pointers[index]);
	free(state->entries);
}

static void Glm5NextBindLayer(const Glm5NextModelState *state, const SparkStageRunnerStep *step, uint32_t layer, Glm5NextLayerBuffers *b)
{
	const Glm5NextLayerWeights *w = &state->layers[layer];
	const uint32_t D = state->tp_degree, heads = Glm5NextRankHeads(G5N_HEADS, D);
	memset(b, 0, sizeof(*b));
	b->w = w;
	b->layer_index = layer;
	b->tp_rank = state->tp_rank;
	b->attn_heads = heads;
	b->q_b_rows = heads * G5N_NOPE;
	b->attn_output_columns = heads * G5N_VALUE;
	b->dense_gate_up_rows = 2u * G5N_DENSE_INTER / D;
	b->dense_intermediate = G5N_DENSE_INTER / D;
	b->expert_w1_rows = 2u * G5N_EXPERT_INTER / D;
	b->expert_intermediate = G5N_EXPERT_INTER / D;
	b->shared_gate_up_rows = 2u * G5N_EXPERT_INTER / D;
	b->shared_intermediate = G5N_EXPERT_INTER / D;
	b->kda_heads = Glm5NextRankHeads(G5N_KDA_HEADS, D);
	if ( w->kv_b_key != 0 )
		b->kv_b_key = (const uint16_t *)w->kv_b_key + (uint64_t)state->tp_rank * heads * G5N_LATENT * G5N_NOPE;
	if ( w->kv_b_value != 0 )
		b->kv_b_value = (const uint16_t *)w->kv_b_value + (uint64_t)state->tp_rank * heads * G5N_VALUE * G5N_LATENT;
	b->dense_row_offset = state->dense_row_offset;
	b->dense_tile_prefix = state->dense_tile_prefix;
	b->hidden_bf16 = state->streams;
	b->normed_bf16 = state->normed;
	b->q_compressed_bf16 = state->q_compressed;
	b->q_bf16 = state->q;
	b->query_latent_bf16 = state->query_latent;
	b->index_query_bf16 = state->index_query;
	b->index_key_bf16 = state->index_key;
	b->index_head_weight_bf16 = state->index_head_weight;
	b->index_gate_bf16 = state->index_gate;
	b->index_packed_bf16 = state->index_packed;
	b->kv_slot_bf16 = state->kv_slot;
	b->attention_latent_bf16 = state->attention_latent;
	b->attention_value_bf16 = state->attention_value;
	b->attention_out_bf16 = state->attention_out;
	b->gate_up_bf16 = state->gate_up;
	b->intermediate_bf16 = state->intermediate;
	b->expert_out_bf16 = state->expert_out;
	b->shared_out_bf16 = state->shared_out;
	b->router_logits = state->router_logits;
	b->selection_scores = state->selection_scores;
	b->selected_pools = state->selected_pools;
	b->selected_positions = state->selected_positions;
	b->route_expert = state->route_expert;
	b->route_weight = state->route_weight;
	b->route_source_token = state->route_source_token;
	b->route_packed_row = state->route_packed_row;
	b->group_row_offset = state->group_row_offset;
	b->group_tile_prefix_w1 = state->group_tile_prefix_w1;
	b->group_tile_prefix_w2 = state->group_tile_prefix_w2;
	b->fused_qkvb_bf16 = state->fused_qkvb;
	b->fused_decay_gate_bf16 = state->fused_decay_gate;
	b->kda_decay_latent_bf16 = state->kda_decay_latent;
	b->kda_beta_logit = state->kda_beta_logit;
	b->kda_gate_bf16 = state->kda_gate;
	b->kda_gate_latent_bf16 = state->kda_gate_latent;
	b->kda_decay_logit_bf16 = state->kda_decay_logit;
	b->kda_retention = state->kda_retention;
	b->kda_write_gate = state->kda_write_gate;
	b->hc_mixes_f32 = state->hc_mixes;
	b->hc_pre_f32 = state->hc_pre;
	b->hc_post_f32 = state->hc_post;
	b->hc_comb_f32 = state->hc_comb;
	b->hc_collapsed_bf16 = state->collapsed;
	b->hc_snapshot_bf16 = state->snapshot;
	b->shard_query_gathered_bf16 = state->shard_query_gathered;
	b->shard_partials_f32 = state->shard_partials;
	b->shard_partials_received_f32 = state->shard_partials_received;
	b->index_owner_rank = state->tp_rank;
	b->index_owner_degree = state->tp_degree;
	b->index_local_scores = state->index_local_scores;
	b->index_gathered_scores = state->index_gathered_scores;
	b->sequence_of_row = step->sequence_of_row;
	b->context_length = step->context_length;
	b->positions = step->positions;
	b->kda_state_index = step->recurrent_index;
	b->sequence_row_begin = step->sequence_row_begin;
	b->sequence_row_indices = step->sequence_row_indices;
	b->shard_query_stride = SparkGlm5NextKvShardQueryStride(step->rows, D);
	b->shard_partial_stride = SparkGlm5NextKvShardPartialStride(step->rows, D, UINT32_MAX);
	if ( w->kda_ordinal != G5N_NO_ORDINAL )
	{
		const uint64_t windows = (uint64_t)G5N_KDA_LAYERS * state->slots * state->kda_window_slot_bytes;
		const uint64_t window = (uint64_t)w->kda_ordinal * state->slots * state->kda_window_slot_bytes;
		b->kda_state_pool = state->kda_state + (uint64_t)w->kda_ordinal * state->slots * state->kda_state_slot_bytes;
		b->kda_state_slot_bytes = (uint32_t)state->kda_state_slot_bytes;
		b->kda_q_window = (uint16_t *)(state->kda_windows + window);
		b->kda_k_window = (uint16_t *)(state->kda_windows + windows + window);
		b->kda_v_window = (uint16_t *)(state->kda_windows + 2u * windows + window);
	}
	if ( w->dsa_ordinal != G5N_NO_ORDINAL )
	{
		b->cache = state->latent_views[w->dsa_ordinal];
		b->index_cache = state->index_views[w->dsa_ordinal];
	}
}

template<class G>
struct Glm5NextStage
{
	static void Fail(Glm5NextModelState *state, SparkStatus status)
	{
		state->services->fail(state->services->context, status);
	}

	static SparkStatus Round(Glm5NextModelState *state, cudaStream_t stream, uint32_t operation, uint32_t rows, const void *local, void *full)
	{
		if ( rows == 0u || rows > state->collective_capacity )
		{
			fprintf(stderr, "%s: a %u-sequence collective exceeds the %u the collective was sized for\n", G::Tag(), rows, state->collective_capacity);
			SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
		}
		return state->services->round(state->services->context, (void *)stream, operation, SPARK_STAGE_RUNNER_ROUND_ALL, rows, 0u, local, full);
	}

	static void RowState(const Glm5NextModelState *state, uint32_t layer, uint32_t phase)
	{
		if ( state->services->row_state != 0 )
			state->services->row_state(state->services->context, layer, phase);
	}

	static SparkStatus Launch(int32_t status, const char *site, uint32_t layer)
	{
		if ( status == LM_LAUNCH_OK )
			return SPARK_STATUS_OK;
		fprintf(stderr, "%s: %s failed at layer %u status=%d cuda=%s\n", G::Tag(), site, layer, (int)status, cudaGetErrorString(cudaPeekAtLastError()));
		SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
	}

	static SparkStatus Attention(Glm5NextModelState *state, const Glm5NextLayerBuffers *b, const SparkStageRunnerStep *step, uint32_t context,
		cudaStream_t stream)
	{
		const uint32_t rows = step->rows, sms = step->multiprocessors, layer = b->layer_index, D = state->tp_degree;
		SparkStatus status;
		RowState(state, layer, SPARK_STAGE_RUNNER_ROW_STATE_READ);
		status = Launch(Glm5NextAttentionHead(b, rows, context, sms, stream), "attention head", layer);
		if ( status == SPARK_STATUS_OK )
			status = Round(state, stream, SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_GATHER,
				SparkGlm5NextIndexCpGatherSequences(rows, Glm5NextLocalStride(state, context)), b->index_local_scores, (void *)b->index_gathered_scores);
		if ( status == SPARK_STATUS_OK )
			status = Launch(Glm5NextAttentionTail(b, rows, context, sms, stream), "attention tail", layer);
		if ( status == SPARK_STATUS_OK )
			status = Round(state, stream, SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_GATHER, SparkGlm5NextKvShardQuerySequences(rows, D),
				b->query_latent_bf16, b->shard_query_gathered_bf16);
		if ( status == SPARK_STATUS_OK )
			status = Launch(Glm5NextAttentionPartial(b, rows, stream), "attention partial", layer);
		RowState(state, layer, SPARK_STAGE_RUNNER_ROW_STATE_WRITTEN);
		if ( status == SPARK_STATUS_OK )
			status = Round(state, stream, SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_TO_ALL, SparkGlm5NextKvShardPartialSequences(rows, D, UINT32_MAX),
				b->shard_partials_f32, b->shard_partials_received_f32);
		if ( status == SPARK_STATUS_OK )
			status = Launch(Glm5NextAttentionMerge(b, rows, sms, stream), "attention merge", layer);
		return status;
	}

	static SparkStatus Mlp(Glm5NextModelState *state, Glm5NextLayerBuffers *b, const SparkStageRunnerStep *step, cudaStream_t stream)
	{
		const uint32_t rows = step->rows, sms = step->multiprocessors, layer = b->layer_index;
		const void *w1 = 0, *w2 = 0;
		SparkStatus status;
		if ( layer < G5N_FIRST_ROUTED )
			return Launch(Glm5NextDenseMlp(b, rows, sms, stream), "dense mlp", layer);
		status = Launch(Glm5NextMoeRoute(b, rows, sms, stream), "moe route", layer);
		if ( status == SPARK_STATUS_OK )
			status = state->services->expert_weights(state->services->context, layer, b->group_row_offset, rows * G5N_TOP_K, &w1, &w2);
		if ( status != SPARK_STATUS_OK )
			return status;
		b->expert_w1_weight = w1;
		b->expert_w2_weight = w2;
		if ( LmWeightCodec<G::kExpertCodec>::kScaleEncoding != LM_SCALE_ENCODING_NONE )
		{
			b->expert_w1_scale = (const uint8_t *)w1 - b->w->expert_w1_offset + b->w->expert_w1_scale_offset;
			b->expert_w2_scale = (const uint8_t *)w2 - b->w->expert_w2_offset + b->w->expert_w2_scale_offset;
		}
		status = Launch(Glm5NextMoeExperts<G::kExpertCodec>(b, rows, sms, stream), "moe experts", layer);
		state->services->layer_done(state->services->context, layer);
		return status;
	}

	static SparkStatus Step(void *model, const SparkStageRunnerStep *step, void *stream_void)
	{
		Glm5NextModelState *state = (Glm5NextModelState *)model;
		cudaStream_t stream = (cudaStream_t)stream_void;
		const uint32_t rows = step->rows;
		const uint32_t context = step->gather_context != 0u && step->gather_context <= state->context_capacity ? step->gather_context : state->context_capacity;
		Glm5NextLayerBuffers b;
		SparkStatus status = SPARK_STATUS_OK;
		if ( state->kv_attached == 0u || rows == 0u || rows > state->max_rows || step->sequences == 0u || step->sequences > state->slots ||
			step->recurrent_index == 0 || step->sequence_row_begin == 0 || step->sequence_row_indices == 0 )
		{
			fprintf(stderr, "%s: step refused rows=%u sequences=%u kv_attached=%u\n", G::Tag(), rows, step->sequences, state->kv_attached);
			SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
		}
		Glm5NextDenseOffsetsKernel<<<1u, 1u, 0, stream>>>(state->dense_row_offset, rows);
		Glm5NextExpandStreamsKernel<<<dim3((G5N_HIDDEN + G5N_THREADS - 1u) / G5N_THREADS, rows), G5N_THREADS, 0, stream>>>(step->hidden_bf16,
			state->streams, rows);
		if ( cudaPeekAtLastError() != cudaSuccess )
			SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
		for ( uint32_t layer = 0u; layer < G5N_LAYERS && status == SPARK_STATUS_OK; ++layer )
		{
			const Glm5NextLayerWeights *w = &state->layers[layer];
			Glm5NextBindLayer(state, step, layer, &b);
			status = Launch(Glm5NextHcSite(&b, w->hc_attn_fn, w->hc_attn_base, w->hc_attn_scale, rows, stream), "attention hc site", layer);
			if ( status == SPARK_STATUS_OK )
			{
				if ( w->kda_ordinal != G5N_NO_ORDINAL )
				{
					RowState(state, layer, SPARK_STAGE_RUNNER_ROW_STATE_READ);
					status = Launch(Glm5NextKda(&b, rows, step->sequences, step->commit, step->multiprocessors, stream), "kda", layer);
					RowState(state, layer, SPARK_STAGE_RUNNER_ROW_STATE_WRITTEN);
				}
				else
					status = Attention(state, &b, step, context, stream);
			}
			if ( status == SPARK_STATUS_OK )
				status = Round(state, stream, SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_REDUCE_SUM_BF16, rows, b.attention_out_bf16, b.attention_out_bf16);
			if ( status == SPARK_STATUS_OK )
				status = Launch(Glm5NextHcPost(&b, rows, stream), "attention hc post", layer);
			if ( status == SPARK_STATUS_OK )
				status = Launch(Glm5NextHcSite(&b, w->hc_ffn_fn, w->hc_ffn_base, w->hc_ffn_scale, rows, stream), "ffn hc site", layer);
			if ( status == SPARK_STATUS_OK )
				status = Mlp(state, &b, step, stream);
			if ( status == SPARK_STATUS_OK )
				status = Round(state, stream, SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_REDUCE_SUM_BF16, rows, b.attention_out_bf16, b.attention_out_bf16);
			if ( status == SPARK_STATUS_OK )
				status = Launch(Glm5NextHcPost(&b, rows, stream), "ffn hc post", layer);
		}
		if ( status != SPARK_STATUS_OK )
			return status;
		Glm5NextStreamsMeanKernel<<<rows, G5N_THREADS, 0, stream>>>(state->streams, state->streams_mean, rows);
		LmBf16RmsNormLaunch<G5N_THREADS>(rows, stream, state->streams_mean,
			(const uint16_t *)state->final_norm, step->hidden_bf16, G5N_HIDDEN, G5N_HIDDEN, G5N_EPS);
		if ( cudaPeekAtLastError() != cudaSuccess )
			SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
		state->steps++;
		if ( rows > step->sequences )
			state->prefill_rows += rows;
		return SPARK_STATUS_OK;
	}

	static SparkStatus Open(const SparkStageRunnerModelOpen *request, void **model, SparkStageRunnerModelGeometry *geometry)
	{
		const SparkStageRunnerConfiguration *configuration = request->configuration;
		Glm5NextModelState *state;
		SparkStatus status;
		*model = 0;
		if ( configuration->stage_count != 1u || configuration->tp_degree != G::kTpDegree || configuration->tp_rank >= configuration->tp_degree ||
			configuration->device_collective == 0 || configuration->kv_pages_per_sequence == 0u ||
			configuration->resident_sequence_capacity > configuration->max_active_sequence_count )
		{
			fprintf(stderr, "%s: the runner must be one TP%u stage with a device collective and KV pages (stages=%u tp=%u)\n", G::Tag(),
				G::kTpDegree, configuration->stage_count, configuration->tp_degree);
			SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
		}
		state = (Glm5NextModelState *)calloc(1u, sizeof(*state));
		if ( state == 0 )
			SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
		*model = state;
		state->services = request->services;
		state->tp_degree = configuration->tp_degree;
		state->tp_rank = configuration->tp_rank;
		state->max_rows = configuration->max_input_row_count;
		state->slots = configuration->max_active_sequence_count;
		state->context_capacity = configuration->kv_pages_per_sequence * G5N_PAGE_SLOTS;
		state->collective_capacity = configuration->device_collective->max_active_sequence_count;
		state->kda_state_slot_bytes = (uint64_t)Glm5NextRankHeads(G5N_KDA_HEADS, state->tp_degree) * G5N_KDA_KEY * G5N_KDA_VALUE * sizeof(float);
		state->kda_window_slot_bytes = (uint64_t)Glm5NextRankHeads(G5N_KDA_HEADS, state->tp_degree) * G5N_KDA_KEY * G5N_KDA_CONV * sizeof(uint16_t);
		for ( uint32_t layer = 0u; layer < G5N_LAYERS; ++layer )
		{
			state->layers[layer].kda_ordinal = Glm5NextKdaOrdinal(layer);
			state->layers[layer].dsa_ordinal = Glm5NextDsaOrdinal(layer);
		}
		if ( SparkGlm5NextKvShardFits(state->tp_degree, state->max_rows) == 0u ||
			SparkGlm5NextIndexCpGatherSequences(state->max_rows, Glm5NextLocalStride(state, state->context_capacity)) > state->collective_capacity ||
			SparkGlm5NextKvShardPartialSequences(state->max_rows, state->tp_degree, UINT32_MAX) > state->collective_capacity ||
			SparkGlm5NextKvShardQuerySequences(state->max_rows, state->tp_degree) > state->collective_capacity ||
			state->max_rows > state->collective_capacity )
		{
			fprintf(stderr, "%s: %u rows with %u-token contexts need more than the %u collective sequences configured\n", G::Tag(),
				state->max_rows, state->context_capacity, state->collective_capacity);
			SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
		}
		memset(geometry, 0, sizeof(*geometry));
		status = Glm5NextPackOpen(state, configuration->rank_pack_path, G::kExpertCodec, geometry);
		if ( status == SPARK_STATUS_OK )
			status = Glm5NextAllocate(state, 0u);
		if ( status == SPARK_STATUS_OK )
			status = Glm5NextLaneAllocate(state);
		if ( status != SPARK_STATUS_OK )
			return status;
		geometry->hidden = G5N_HIDDEN;
		geometry->vocab = G5N_VOCAB;
		geometry->total_layers = G5N_LAYERS;
		geometry->first_layer = 0u;
		geometry->layer_count = G5N_LAYERS;
		geometry->rms_epsilon = G5N_EPS;
		geometry->pack_bytes = state->file_bytes;
		geometry->head_prenormed = 1u;
		geometry->kv_layer_count = G5N_DSA_LAYERS;
		geometry->kv_layer_page_bytes = Glm5NextKv::kPageBytes;
		geometry->kv_second_layer_count = G5N_DSA_LAYERS;
		geometry->kv_second_layer_page_bytes = Glm5NextIndexKv::kPageBytes;
		geometry->kv_shard = SparkGlm5NextKvShardLatent(state->tp_rank, state->tp_degree);
		geometry->recurrent_bytes = (uint64_t)G5N_KDA_LAYERS * (state->kda_state_slot_bytes + 3u * state->kda_window_slot_bytes);
		geometry->experts = G5N_EXPERTS;
		geometry->top_k = G5N_TOP_K;
		geometry->expert_w1_offset = state->expert_w1_offset;
		geometry->expert_w2_offset = state->expert_w2_offset;
		fprintf(stderr, "%s: rank %u/%u rows=%u slots=%u context=%u collective=%u pack=%llu bytes recurrent=%llu bytes per slot\n", G::Tag(),
			state->tp_rank, state->tp_degree, state->max_rows, state->slots, state->context_capacity, state->collective_capacity,
			(unsigned long long)state->file_bytes, (unsigned long long)geometry->recurrent_bytes);
		return SPARK_STATUS_OK;
	}

	static SparkStatus ManifestCheck(void *model, const SparkWeightdManifest *manifest)
	{
		Glm5NextModelState *state = (Glm5NextModelState *)model;
		uint64_t covered = 0u;
		for ( uint32_t index = 0u; index < state->entry_count; ++index )
		{
			const Glm5NextPackEntry *entry = &state->entries[index];
			SparkStatus status;
			if ( entry->tensor_kind != GLM5_NEXT_T_EXPERT_UP_GATE && entry->tensor_kind != GLM5_NEXT_T_EXPERT_DOWN )
				continue;
			status = SparkExpertPlanesCovered(manifest, entry->tensor_kind, entry->layer_index, entry->weight_codec, entry->scale_encoding,
				entry->group_count, entry->payload_offset, entry->payload_bytes, entry->scale_offset, entry->scale_bytes, &covered);
			if ( status != SPARK_STATUS_OK )
			{
				fprintf(stderr, "%s: the .experts manifest does not cover kind %u layer %u of the pack\n", G::Tag(), entry->tensor_kind,
					entry->layer_index);
				return status;
			}
		}
		if ( covered != manifest->range_count )
		{
			fprintf(stderr, "%s: the .experts manifest holds %u ranges, the pack's experts cover %llu\n", G::Tag(), manifest->range_count,
				(unsigned long long)covered);
			SPARK_FAIL(SPARK_STATUS_SCHEMA_ERROR);
		}
		return SPARK_STATUS_OK;
	}

	static SparkStatus Slice(SparkWeightdLazyPack *lazy_pack, const Glm5NextPackEntry *entry, const void **pointer)
	{
		if ( SparkWeightdLazyPackSlice(lazy_pack, entry->payload_offset, entry->payload_bytes, pointer) != SPARK_STATUS_OK || *pointer == 0 )
		{
			fprintf(stderr, "%s: spine tensor kind %u layer %u is not resident in the weightd arena\n", G::Tag(), entry->tensor_kind, entry->layer_index);
			SPARK_FAIL(SPARK_STATUS_PARSE_ERROR);
		}
		return SPARK_STATUS_OK;
	}

	static SparkStatus Bind(void *model, SparkWeightdLazyPack *lazy_pack, SparkStageRunnerModelGeometry *geometry)
	{
		Glm5NextModelState *state = (Glm5NextModelState *)model;
		(void)geometry;
		for ( uint32_t index = 0u; index < state->entry_count; ++index )
		{
			const Glm5NextPackEntry *entry = &state->entries[index];
			const void *p = 0;
			Glm5NextLayerWeights *w;
			if ( entry->tensor_kind == GLM5_NEXT_T_EXPERT_UP_GATE || entry->tensor_kind == GLM5_NEXT_T_EXPERT_DOWN ||
				entry->tensor_kind == GLM5_NEXT_T_EMBEDDING || entry->tensor_kind == GLM5_NEXT_T_LM_HEAD ||
				(entry->layer_index >= G5N_LAYERS && entry->layer_index != GLM5_NEXT_PACK_GLOBAL_LAYER) )
				continue;
			if ( Slice(lazy_pack, entry, &p) != SPARK_STATUS_OK )
				SPARK_FAIL(SPARK_STATUS_PARSE_ERROR);
			if ( entry->tensor_kind == GLM5_NEXT_T_FINAL_NORM )
			{
				state->final_norm = p;
				continue;
			}
			w = &state->layers[entry->layer_index];
			switch ( entry->tensor_kind )
			{
			case GLM5_NEXT_T_ATTN_NORM: w->attn_norm = p; break;
			case GLM5_NEXT_T_Q_A: w->q_a = p; break;
			case GLM5_NEXT_T_Q_A_NORM: w->q_a_norm = p; break;
			case GLM5_NEXT_T_Q_B: w->q_b = p; break;
			case GLM5_NEXT_T_KV_A: w->kv_a = p; break;
			case GLM5_NEXT_T_KV_A_NORM: w->kv_a_norm = p; break;
			case GLM5_NEXT_T_KV_B_KEY_T: w->kv_b_key = p; break;
			case GLM5_NEXT_T_KV_B_VALUE: w->kv_b_value = p; break;
			case GLM5_NEXT_T_ATTN_OUTPUT: w->attn_output = p; break;
			case GLM5_NEXT_T_POST_ATTN_NORM: w->post_attn_norm = p; break;
			case GLM5_NEXT_T_INDEX_Q: w->index_q = p; break;
			case GLM5_NEXT_T_INDEX_K: w->index_k = p; break;
			case GLM5_NEXT_T_INDEX_HEAD: w->index_head = p; break;
			case GLM5_NEXT_T_INDEX_NORM_WEIGHT: w->index_norm_weight = p; break;
			case GLM5_NEXT_T_INDEX_NORM_BIAS: w->index_norm_bias = p; break;
			case GLM5_NEXT_T_INDEX_COMPRESS_APE: w->index_compress_ape = p; break;
			case GLM5_NEXT_T_INDEX_COMPRESS_GATE: w->index_compress_gate = p; break;
			case GLM5_NEXT_T_DENSE_GATE_UP: w->dense_gate_up = p; break;
			case GLM5_NEXT_T_DENSE_DOWN: w->dense_down = p; break;
			case GLM5_NEXT_T_ROUTER: w->router = p; break;
			case GLM5_NEXT_T_ROUTER_CORRECTION: w->router_correction = (const float *)p; break;
			case GLM5_NEXT_T_SHARED_GATE_UP: w->shared_gate_up = p; break;
			case GLM5_NEXT_T_SHARED_DOWN: w->shared_down = p; break;
			case GLM5_NEXT_T_KDA_QKV_BETA: w->kda_qkv_beta = p; break;
			case GLM5_NEXT_T_KDA_DECAY_GATE_DOWN: w->kda_decay_gate_down = p; break;
			case GLM5_NEXT_T_KDA_DECAY_UP: w->kda_decay_up = p; break;
			case GLM5_NEXT_T_KDA_GATE_UP: w->kda_gate_up = p; break;
			case GLM5_NEXT_T_KDA_Q_CONV: w->kda_q_conv = p; break;
			case GLM5_NEXT_T_KDA_K_CONV: w->kda_k_conv = p; break;
			case GLM5_NEXT_T_KDA_V_CONV: w->kda_v_conv = p; break;
			case GLM5_NEXT_T_KDA_DECAY_BIAS: w->kda_decay_bias = (const float *)p; break;
			case GLM5_NEXT_T_KDA_HEAD_LOG_SCALE: w->kda_head_log_scale = (const float *)p; break;
			case GLM5_NEXT_T_KDA_OUT_NORM: w->kda_out_norm = p; break;
			case GLM5_NEXT_T_KDA_OUT: w->kda_out = p; break;
			case GLM5_NEXT_T_HC_ATTN_FN: w->hc_attn_fn = p; break;
			case GLM5_NEXT_T_HC_ATTN_BASE: w->hc_attn_base = p; break;
			case GLM5_NEXT_T_HC_ATTN_SCALE: w->hc_attn_scale = p; break;
			case GLM5_NEXT_T_HC_FFN_FN: w->hc_ffn_fn = p; break;
			case GLM5_NEXT_T_HC_FFN_BASE: w->hc_ffn_base = p; break;
			case GLM5_NEXT_T_HC_FFN_SCALE: w->hc_ffn_scale = p; break;
			default:
				fprintf(stderr, "%s: tensor kind %u at layer %u has no place in the stage\n", G::Tag(), entry->tensor_kind, entry->layer_index);
				SPARK_FAIL(SPARK_STATUS_SCHEMA_ERROR);
			}
		}
		if ( state->final_norm == 0 )
			SPARK_FAIL(SPARK_STATUS_PARSE_ERROR);
		return SPARK_STATUS_OK;
	}

	static void LaneFold(Glm5NextModelState *state)
	{
		for ( uint32_t lane = 0u; lane < SPARK_STAGE_RUNNER_LANES; ++lane )
		{
			const Glm5NextModelState *copy = state->lane_states[lane];
			if ( state->lane_bound[lane] == 0u || copy == 0 )
				continue;
			state->steps += copy->steps - state->lane_base[lane][0];
			state->prefill_rows += copy->prefill_rows - state->lane_base[lane][1];
			state->lane_base[lane][0] = copy->steps;
			state->lane_base[lane][1] = copy->prefill_rows;
		}
	}

	static SparkStatus Lane(void *model, uint32_t lane, const SparkStageRunnerStep *step, const SparkStageRunnerServices *services, void **lane_model)
	{
		Glm5NextModelState *state = (Glm5NextModelState *)model;
		Glm5NextModelState *copy;
		void **fields[G5N_SCRATCH_FIELDS], **donor_fields[G5N_SCRATCH_FIELDS];
		if ( lane >= SPARK_STAGE_RUNNER_LANES || step == 0 || services == 0 || lane_model == 0 || state->lane_states[lane] == 0 ||
			state->lane_scratch == 0 )
			SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
		if ( step->rows > state->max_rows )
			return SPARK_STATUS_UNSUPPORTED;
		LaneFold(state);
		copy = state->lane_states[lane];
		memcpy(copy, state, sizeof(*copy));
		copy->services = services;
		if ( lane != 0u )
		{
			const uint32_t count = Glm5NextScratch(copy, fields);
			if ( Glm5NextScratch(state->lane_scratch, donor_fields) != count )
				SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
			for ( uint32_t index = 0u; index < count; ++index )
				*fields[index] = *donor_fields[index];
		}
		state->lane_bound[lane] = 1u;
		state->lane_base[lane][0] = copy->steps;
		state->lane_base[lane][1] = copy->prefill_rows;
		*lane_model = copy;
		return SPARK_STATUS_OK;
	}

	static void Close(void *model)
	{
		Glm5NextModelState *state = (Glm5NextModelState *)model;
		if ( state == 0 )
			return;
		Glm5NextFree(state);
		free(state);
	}

	static SparkStatus Sideband(void *model, uint32_t to_buffer, const void *hidden, void *buffer, uint64_t bytes, uint32_t rows, void *stream)
	{
		(void)model; (void)to_buffer; (void)hidden; (void)buffer; (void)bytes; (void)rows; (void)stream;
		fprintf(stderr, "%s: the TP%u stage owns every layer; a pipeline boundary sideband is not part of this deployment\n", G::Tag(), G::kTpDegree);
		SPARK_FAIL(SPARK_STATUS_UNSUPPORTED);
	}

	static SparkStatus AttachKv(void *model, const SparkStageRunnerKv *kv)
	{
		Glm5NextModelState *state = (Glm5NextModelState *)model;
		const SparkKvShard latent = SparkGlm5NextKvShardLatent(state->tp_rank, state->tp_degree);
		const SparkKvShard index = SparkGlm5NextKvShardIndex(state->tp_rank, state->tp_degree);
		if ( kv->layer_count != G5N_DSA_LAYERS || kv->second_layer_count != G5N_DSA_LAYERS || kv->second_pool == 0 ||
			kv->layer_page_bytes != Glm5NextKv::kPageBytes || kv->second_layer_page_bytes != Glm5NextIndexKv::kPageBytes ||
			kv->context_shard.degree != latent.degree || kv->context_shard.rank != latent.rank || kv->context_shard.grain != latent.grain ||
			kv->sequence_count > state->slots )
		{
			fprintf(stderr, "%s: KV attach refused: %u+%u layers, shard %u/%u grain %u, %u sequences\n", G::Tag(), kv->layer_count,
				kv->second_layer_count, kv->context_shard.rank, kv->context_shard.degree, kv->context_shard.grain, kv->sequence_count);
			SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
		}
		for ( uint32_t layer = 0u; layer < G5N_DSA_LAYERS; ++layer )
			if ( LmKvShardViewInitialize<Glm5NextKv>(&state->latent_views[layer], kv->pool + (uint64_t)layer * kv->layer_stride_bytes,
					kv->page_table, kv->page_table_stride, kv->sequence_count, kv->pool_page_count, state->kv_error, latent) != 0 ||
				LmKvShardViewInitialize<Glm5NextIndexKv>(&state->index_views[layer], kv->second_pool + (uint64_t)layer * kv->second_layer_stride_bytes,
					kv->page_table, kv->page_table_stride, kv->sequence_count, kv->pool_page_count, state->kv_error, index) != 0 )
				SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
		state->kv_attached = 1u;
		return SPARK_STATUS_OK;
	}

	static SparkStatus RecurrentCopy(void *model, uint32_t to_buffer, uint32_t slot, void *buffer, uint64_t bytes, void *stream)
	{
		Glm5NextModelState *state = (Glm5NextModelState *)model;
		const uint64_t windows = (uint64_t)G5N_KDA_LAYERS * state->slots * state->kda_window_slot_bytes;
		uint8_t *pools[4] = { state->kda_state, state->kda_windows, state->kda_windows + windows, state->kda_windows + 2u * windows };
		const uint64_t widths[4] = { state->kda_state_slot_bytes, state->kda_window_slot_bytes, state->kda_window_slot_bytes, state->kda_window_slot_bytes };
		uint8_t *packed = (uint8_t *)buffer;
		cudaError_t error = cudaSuccess;
		if ( slot >= state->slots || buffer == 0 || bytes != (uint64_t)G5N_KDA_LAYERS * (widths[0] + 3u * widths[1]) )
			SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
		for ( uint32_t part = 0u; part < 4u && error == cudaSuccess; ++part )
		{
			uint8_t *pool = pools[part] + (uint64_t)slot * widths[part];
			const size_t pitch = (size_t)(widths[part] * state->slots);
			error = to_buffer != 0u ?
				cudaMemcpy2DAsync(packed, (size_t)widths[part], pool, pitch, (size_t)widths[part], G5N_KDA_LAYERS, cudaMemcpyDefault, (cudaStream_t)stream) :
				cudaMemcpy2DAsync(pool, pitch, packed, (size_t)widths[part], (size_t)widths[part], G5N_KDA_LAYERS, cudaMemcpyDefault, (cudaStream_t)stream);
			packed += widths[part] * G5N_KDA_LAYERS;
		}
		if ( error == cudaSuccess && stream == 0 )
			error = cudaStreamSynchronize(0);
		if ( error != cudaSuccess )
		{
			fprintf(stderr, "%s: recurrent copy failed slot=%u cuda=%s\n", G::Tag(), slot, cudaGetErrorString(error));
			SPARK_FAIL(SPARK_STATUS_IO_ERROR);
		}
		return SPARK_STATUS_OK;
	}

	static SparkStatus ResetSlot(void *model, uint32_t slot, void *stream)
	{
		Glm5NextModelState *state = (Glm5NextModelState *)model;
		const uint64_t windows = (uint64_t)G5N_KDA_LAYERS * state->slots * state->kda_window_slot_bytes;
		uint8_t *pools[4] = { state->kda_state, state->kda_windows, state->kda_windows + windows, state->kda_windows + 2u * windows };
		const uint64_t widths[4] = { state->kda_state_slot_bytes, state->kda_window_slot_bytes, state->kda_window_slot_bytes, state->kda_window_slot_bytes };
		cudaError_t error = cudaSuccess;
		if ( slot >= state->slots )
			SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
		for ( uint32_t part = 0u; part < 4u && error == cudaSuccess; ++part )
			error = cudaMemset2DAsync(pools[part] + (uint64_t)slot * widths[part], (size_t)(widths[part] * state->slots), 0, (size_t)widths[part],
				G5N_KDA_LAYERS, (cudaStream_t)stream);
		return error == cudaSuccess ? SPARK_STATUS_OK : SPARK_STATUS_IO_ERROR;
	}

	static const void *ProbeBuffers(void *model)
	{
		return model;
	}

	static void Report(void *model, uint32_t rank)
	{
		Glm5NextModelState *state = (Glm5NextModelState *)model;
		LaneFold(state);
		fprintf(stderr, "%s rank=%u steps=%llu prefill_rows=%llu context_capacity=%u\n", G::ReportTag(), rank, (unsigned long long)state->steps,
			(unsigned long long)state->prefill_rows, state->context_capacity);
	}

	static const SparkStageRunnerModelInterface *Interface(void)
	{
		static const SparkStageRunnerModelInterface table =
		{
			SPARK_STAGE_RUNNER_MODEL_ABI_VERSION,
			G::Tag(),
			G::WeightdModel(),
			G::WeightdRevision(),
			Open,
			ManifestCheck,
			Bind,
			Close,
			Step,
			Sideband,
			AttachKv,
			RecurrentCopy,
			ResetSlot,
			ProbeBuffers,
			Report,
			0,
			Lane
		};
		return &table;
	}
};
