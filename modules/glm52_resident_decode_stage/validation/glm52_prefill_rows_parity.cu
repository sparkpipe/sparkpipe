#define main SparkGlm52ValidatorMain
#include "spark_glm52_resident_decode_stage_cuda_validation.cu"
#undef main

#include "sparkpipe/spark_glm52_graph_regime.h"
#include "sparkpipe/spark_row_layout.h"

#define ROWS_MAX 16u
#define ROWS_LAYERS 2u
#define ROWS_FIRST_LAYER (SPARK_GLM52_MODEL_FIRST_ROUTED_LAYER - 1u)
#define ROWS_SPLIT_THRESHOLD 64u
#define ROWS_BOUNDARY (2u * SPARK_GLM52_VHIDDEN)
#define ROWS_VERIFY_DEPTH 8u
#define ROWS_ADVERSARY_TAIL 4u

typedef struct RowsRig
{
	SparkGlm52ValFixture fixture;
	SparkGlm52LayerWeights layers[ROWS_LAYERS];
	uint32_t ordinals[ROWS_LAYERS];
	uint32_t positions_total;
	uint32_t *host_tokens;
	uint32_t *host_slots;
	uint32_t *host_positions;
	uint16_t *boundary;
	uint16_t *reference_boundary;
	uint16_t *mode_boundary;
	uint8_t *reference_kv;
	uint8_t *mode_kv;
	uint8_t *reference_index;
	uint8_t *mode_index;
	uint8_t *kv_cache;
	uint8_t *index_cache;
	uint32_t *page_table;
	uint64_t kv_bytes;
	uint64_t index_bytes;
	uint64_t waves;
} RowsRig;

typedef struct RowsRegimeContext
{
	const uint32_t *positions;
} RowsRegimeContext;

static uint32_t RowsToken(uint32_t position)
{
	return((position * 5u + 3u + (position / 7u)) % SPARK_GLM52_VALIDATION_EMBED_ROWS);
}

static void *RowsAlloc(uint64_t bytes)
{
	return(SparkGlm52ValAllocZeroed(bytes));
}

static int RowsSetup(RowsRig *rig,uint32_t positions_total)
{
	SparkGlm52ValFixture *fixture = &rig->fixture;
	SparkGlm52ExecutionSlot *slot = &fixture->slot;
	uint32_t pages,lane,page;
	uint32_t *host_table;
	rig->positions_total = positions_total;
	pages = (positions_total + SPARK_GLM52_VKV_PAGE_SLOTS - 1u) / SPARK_GLM52_VKV_PAGE_SLOTS;
	rig->kv_bytes = (uint64_t)pages * SPARK_GLM52_VKV_PAGE_BYTES;
	rig->index_bytes = (uint64_t)pages * SPARK_GLM52_VINDEX_PAGE_BYTES;
	rig->kv_cache = (uint8_t *)RowsAlloc(rig->kv_bytes);
	rig->index_cache = (uint8_t *)RowsAlloc(rig->index_bytes);
	rig->page_table = (uint32_t *)RowsAlloc((uint64_t)ROWS_MAX * pages * sizeof(uint32_t));
	rig->boundary = (uint16_t *)RowsAlloc((uint64_t)positions_total * ROWS_BOUNDARY * sizeof(uint16_t));
	rig->reference_boundary = (uint16_t *)calloc((uint64_t)positions_total * ROWS_BOUNDARY,sizeof(uint16_t));
	rig->mode_boundary = (uint16_t *)calloc((uint64_t)positions_total * ROWS_BOUNDARY,sizeof(uint16_t));
	rig->reference_kv = (uint8_t *)calloc(rig->kv_bytes,1u);
	rig->mode_kv = (uint8_t *)calloc(rig->kv_bytes,1u);
	rig->reference_index = (uint8_t *)calloc(rig->index_bytes,1u);
	rig->mode_index = (uint8_t *)calloc(rig->index_bytes,1u);
	host_table = (uint32_t *)calloc((uint64_t)ROWS_MAX * pages,sizeof(uint32_t));
	if ( cudaHostAlloc((void **)&rig->host_tokens,3u * ROWS_MAX * sizeof(uint32_t),cudaHostAllocPortable) != cudaSuccess )
		return(1);
	rig->host_slots = rig->host_tokens + ROWS_MAX;
	rig->host_positions = rig->host_slots + ROWS_MAX;
	if ( rig->kv_cache == 0 || rig->index_cache == 0 || rig->page_table == 0 || rig->boundary == 0 || rig->reference_boundary == 0 || rig->mode_boundary == 0 ||
		rig->reference_kv == 0 || rig->mode_kv == 0 || rig->reference_index == 0 || rig->mode_index == 0 || host_table == 0 )
		return(1);
	for (lane=0u; lane<ROWS_MAX; lane++)
		for (page=0u; page<pages; page++)
			host_table[lane * pages + page] = page;
	if ( cudaMemcpy(rig->page_table,host_table,(uint64_t)ROWS_MAX * pages * sizeof(uint32_t),cudaMemcpyHostToDevice) != cudaSuccess )
		return(1);
	free(host_table);
	SparkGlm52ValBuildWave(fixture,ROWS_FIRST_LAYER,0u,0u);
	slot->hidden_bf16 = (uint16_t *)RowsAlloc((uint64_t)SPARK_GLM52_VHIDDEN * ROWS_MAX * sizeof(uint16_t));
	slot->residual_bf16 = (uint16_t *)RowsAlloc((uint64_t)SPARK_GLM52_VHIDDEN * ROWS_MAX * sizeof(uint16_t));
	slot->normed_bf16 = (uint16_t *)RowsAlloc((uint64_t)SPARK_GLM52_VHIDDEN * ROWS_MAX * sizeof(uint16_t));
	slot->q_compressed_bf16 = (uint16_t *)RowsAlloc((uint64_t)SPARK_GLM52_VQUERY_A * ROWS_MAX * sizeof(uint16_t));
	slot->q_bf16 = (uint16_t *)RowsAlloc((uint64_t)SPARK_GLM52_VQ_ROWS * ROWS_MAX * sizeof(uint16_t));
	slot->query_latent_bf16 = (uint16_t *)RowsAlloc((uint64_t)SPARK_GLM52_VHEADS * SPARK_GLM52_VLATENT * ROWS_MAX * sizeof(uint16_t));
	slot->query_rope_bf16 = (uint16_t *)RowsAlloc((uint64_t)SPARK_GLM52_VHEADS * SPARK_GLM52_VROPE * ROWS_MAX * sizeof(uint16_t));
	slot->index_query_bf16 = (uint16_t *)RowsAlloc((uint64_t)SPARK_GLM52_VDSA_QUERY_DIM * ROWS_MAX * sizeof(uint16_t));
	slot->index_key_bf16 = (uint16_t *)RowsAlloc((uint64_t)SPARK_GLM52_VDSA_DIM * ROWS_MAX * sizeof(uint16_t));
	slot->index_head_weight_bf16 = (uint16_t *)RowsAlloc((uint64_t)SPARK_GLM52_VDSA_HEADS * ROWS_MAX * sizeof(uint16_t));
	slot->kv_slot_bf16 = (uint16_t *)RowsAlloc((uint64_t)SPARK_GLM52_VKV_SLOT_ELEMENTS * ROWS_MAX * sizeof(uint16_t));
	slot->attention_latent_bf16 = (uint16_t *)RowsAlloc((uint64_t)SPARK_GLM52_VHEADS * SPARK_GLM52_VLATENT * ROWS_MAX * sizeof(uint16_t));
	slot->attention_value_bf16 = (uint16_t *)RowsAlloc((uint64_t)SPARK_GLM52_VHEADS * SPARK_GLM52_VVALUE_DIM * ROWS_MAX * sizeof(uint16_t));
	slot->attention_out_bf16 = (uint16_t *)RowsAlloc((uint64_t)SPARK_GLM52_VHIDDEN * ROWS_MAX * sizeof(uint16_t));
	slot->gate_up_bf16 = (uint16_t *)RowsAlloc((uint64_t)SPARK_GLM52_VGATE_UP_ROWS * ROWS_MAX * SPARK_GLM52_VTOP_K * sizeof(uint16_t));
	slot->intermediate_bf16 = (uint16_t *)RowsAlloc((uint64_t)SPARK_GLM52_VGATE_UP_ROWS * ROWS_MAX * SPARK_GLM52_VTOP_K * sizeof(uint16_t));
	slot->expert_out_bf16 = (uint16_t *)RowsAlloc((uint64_t)ROWS_MAX * SPARK_GLM52_VTOP_K * SPARK_GLM52_VHIDDEN * sizeof(uint16_t));
	slot->shared_out_bf16 = (uint16_t *)RowsAlloc((uint64_t)ROWS_MAX * SPARK_GLM52_VHIDDEN * sizeof(uint16_t));
	slot->router_logits_f32 = (float *)RowsAlloc((uint64_t)ROWS_MAX * 256u * sizeof(float));
	slot->selection_scores_f32 = (float *)RowsAlloc((uint64_t)ROWS_MAX * positions_total * sizeof(float));
	slot->selected_positions = (uint32_t *)RowsAlloc((uint64_t)ROWS_MAX * SPARK_GLM52_VDSA_SELECTED * sizeof(uint32_t));
	slot->route_expert = (uint32_t *)RowsAlloc((uint64_t)ROWS_MAX * SPARK_GLM52_VTOP_K * sizeof(uint32_t));
	slot->route_weight = (float *)RowsAlloc((uint64_t)ROWS_MAX * SPARK_GLM52_VTOP_K * sizeof(float));
	slot->route_source_token = (uint32_t *)RowsAlloc((uint64_t)ROWS_MAX * SPARK_GLM52_VTOP_K * sizeof(uint32_t));
	slot->route_packed_row = (uint32_t *)RowsAlloc((uint64_t)ROWS_MAX * SPARK_GLM52_VTOP_K * sizeof(uint32_t));
	slot->group_row_offset = (uint32_t *)RowsAlloc(1024u * sizeof(uint32_t));
	slot->group_tile_prefix_w1 = (uint32_t *)RowsAlloc(1024u * sizeof(uint32_t));
	slot->group_tile_prefix_w2 = (uint32_t *)RowsAlloc(1024u * sizeof(uint32_t));
	slot->dense_row_offset = (uint32_t *)RowsAlloc(64u * sizeof(uint32_t));
	slot->dense_tile_prefix = (uint32_t *)RowsAlloc(1024u * sizeof(uint32_t));
	slot->context_lengths = (uint32_t *)RowsAlloc(ROWS_MAX * sizeof(uint32_t));
	slot->token_ids = (uint32_t *)RowsAlloc(ROWS_MAX * sizeof(uint32_t));
	slot->resident_slots = (uint32_t *)RowsAlloc(ROWS_MAX * sizeof(uint32_t));
	slot->positions = (uint32_t *)RowsAlloc(ROWS_MAX * sizeof(uint32_t));
	slot->head_maxloc_u64 = (uint64_t *)RowsAlloc(ROWS_MAX * sizeof(uint64_t));
	fixture->split_partials = (float *)RowsAlloc(SPARK_GLM52_RESIDENT_DECODE_STAGE_ATTN_SPLIT_PARTIAL_BYTES(ROWS_MAX,SPARK_GLM52_MODEL_HEAD_COUNT));
	if ( slot->hidden_bf16 == 0 || slot->residual_bf16 == 0 || slot->normed_bf16 == 0 || slot->q_compressed_bf16 == 0 || slot->q_bf16 == 0 ||
		slot->query_latent_bf16 == 0 || slot->query_rope_bf16 == 0 || slot->index_query_bf16 == 0 || slot->index_key_bf16 == 0 ||
		slot->index_head_weight_bf16 == 0 || slot->kv_slot_bf16 == 0 || slot->attention_latent_bf16 == 0 || slot->attention_value_bf16 == 0 ||
		slot->attention_out_bf16 == 0 || slot->gate_up_bf16 == 0 || slot->intermediate_bf16 == 0 || slot->expert_out_bf16 == 0 ||
		slot->shared_out_bf16 == 0 || slot->router_logits_f32 == 0 || slot->selection_scores_f32 == 0 || slot->selected_positions == 0 ||
		slot->route_expert == 0 || slot->route_weight == 0 || slot->route_source_token == 0 || slot->route_packed_row == 0 ||
		slot->group_row_offset == 0 || slot->group_tile_prefix_w1 == 0 || slot->group_tile_prefix_w2 == 0 || slot->dense_row_offset == 0 ||
		slot->dense_tile_prefix == 0 || slot->context_lengths == 0 || slot->token_ids == 0 || slot->resident_slots == 0 || slot->positions == 0 ||
		slot->head_maxloc_u64 == 0 || fixture->split_partials == 0 )
		return(1);
	return(0);
}

static void RowsWave(RowsRig *rig,uint32_t first_position,uint32_t rows,const uint32_t *tokens)
{
	SparkGlm52ValFixture *fixture = &rig->fixture;
	SparkGlm52CudaWave *wave = &fixture->wave;
	uint32_t row,layer,pages;
	pages = (rig->positions_total + SPARK_GLM52_VKV_PAGE_SLOTS - 1u) / SPARK_GLM52_VKV_PAGE_SLOTS;
	for (row=0u; row<rows; row++)
	{
		rig->host_tokens[row] = tokens[row];
		rig->host_slots[row] = 0u;
		rig->host_positions[row] = first_position + row;
	}
	for (layer=0u; layer<ROWS_LAYERS; layer++)
	{
		rig->layers[layer] = fixture->weights;
		rig->ordinals[layer] = 0u;
	}
	wave->first_layer_index = ROWS_FIRST_LAYER;
	wave->layer_count = ROWS_LAYERS;
	wave->layers = rig->layers;
	wave->index_ordinal_by_local_layer = rig->ordinals;
	wave->row_count = rows;
	wave->maximum_context = first_position + rows;
	wave->resident_sequence_capacity = ROWS_MAX;
	wave->max_sequence_positions = rig->positions_total;
	wave->pages_per_sequence = pages;
	wave->host_token_ids = rig->host_tokens;
	wave->host_resident_slots = rig->host_slots;
	wave->host_positions = rig->host_positions;
	wave->owns_embedding = 1u;
	wave->owns_final_head = 0u;
	wave->hidden_output_bf16 = rig->boundary;
	wave->boundary_row_offset = first_position;
	wave->kv_cache = rig->kv_cache;
	wave->kv_layer_stride_bytes = SPARK_GLM52_VKV_LAYER_BYTES;
	wave->index_cache = rig->index_cache;
	wave->index_layer_stride_bytes = 0u;
	wave->page_table = rig->page_table;
	wave->decode_split_context_threshold = ROWS_SPLIT_THRESHOLD;
	wave->attention_split_partials_f32 = fixture->split_partials;
	wave->attention_split_partial_blocks = SPARK_GLM52_RESIDENT_DECODE_STAGE_ATTN_SPLIT_PARTIAL_BLOCKS(ROWS_MAX,SPARK_GLM52_MODEL_HEAD_COUNT);
}

static int RowsWalk(RowsRig *rig)
{
	SparkGlm52CudaWave *wave = &rig->fixture.wave;
	uint32_t layer;
	rig->waves++;
	if ( SparkGlm52LaunchCudaWaveBegin(wave) != 0 )
		return(1);
	for (layer=0u; layer<ROWS_LAYERS; layer++)
	{
		if ( SparkGlm52LaunchCudaLayerAttention(wave,layer) != 0 )
			return(2);
		if ( SparkGlm52LaunchCudaLayerMlp(wave,layer) != 0 )
			return(3);
	}
	if ( SparkGlm52LaunchCudaWaveHead(wave) != 0 )
		return(4);
	if ( cudaStreamSynchronize(rig->fixture.stream) != cudaSuccess )
		return(5);
	return(SparkGlm52ValCheckAccessError(&rig->fixture) != 0 ? 6 : 0);
}

static int RowsReset(RowsRig *rig)
{
	SparkGlm52ExecutionSlot *slot = &rig->fixture.slot;
	if ( cudaMemset(rig->kv_cache,0,rig->kv_bytes) != cudaSuccess ||
		cudaMemset(rig->index_cache,0,rig->index_bytes) != cudaSuccess ||
		cudaMemset(rig->boundary,0,(uint64_t)rig->positions_total * ROWS_BOUNDARY * sizeof(uint16_t)) != cudaSuccess ||
		cudaMemset(slot->context_lengths,0,ROWS_MAX * sizeof(uint32_t)) != cudaSuccess ||
		cudaMemset(slot->selected_positions,0,(uint64_t)ROWS_MAX * SPARK_GLM52_VDSA_SELECTED * sizeof(uint32_t)) != cudaSuccess ||
		cudaMemset(rig->fixture.kv_access_error,0,SPARK_GLM52_VALIDATION_KV_ACCESS_WORDS * sizeof(uint32_t)) != cudaSuccess )
		return(1);
	return(0);
}

static uint32_t RowsRegime(void *context,uint32_t row)
{
	const RowsRegimeContext *regime = (const RowsRegimeContext *)context;
	uint32_t bound,tokens;
	tokens = regime->positions[row] + 1u;
	return(SparkGlm52GraphRegime(tokens,ROWS_SPLIT_THRESHOLD,SPARK_GLM52_MODEL_MAXIMUM_CONTEXT_TOKENS,&bound) +
		(tokens > SPARK_GLM52_MODEL_DSA_SELECTED_TOKEN_COUNT ? SPARK_GLM52_GRAPH_REGIME_COUNT : 0u));
}

static int RowsRun(RowsRig *rig,uint32_t maximum_rows,uint32_t regime_split,uint16_t *boundary_out,uint8_t *kv_out,uint8_t *index_out,uint32_t *waves_out)
{
	SparkRowLayoutDenseLaneContext lanes;
	RowsRegimeContext regime;
	uint32_t *lane_ids,*positions,*tokens,position,rows;
	int status = 0;
	lane_ids = (uint32_t *)calloc(rig->positions_total,sizeof(uint32_t));
	positions = (uint32_t *)calloc(rig->positions_total,sizeof(uint32_t));
	tokens = (uint32_t *)calloc(rig->positions_total,sizeof(uint32_t));
	if ( lane_ids == 0 || positions == 0 || tokens == 0 || RowsReset(rig) != 0 )
		return(100);
	for (position=0u; position<rig->positions_total; position++)
	{
		positions[position] = position;
		tokens[position] = RowsToken(position);
	}
	lanes.lane_count = 1u;
	regime.positions = positions;
	*waves_out = 0u;
	for (position=0u; status==0 && position<rig->positions_total; position+=rows)
	{
		if ( maximum_rows <= 1u )
			rows = 1u;
		else
			rows = SparkRowLayoutRoundSpanWaveRowCount(position,rig->positions_total,lane_ids,SparkRowLayoutDenseLaneOrdinal,&lanes,regime_split != 0u ? RowsRegime : 0,&regime,maximum_rows);
		if ( maximum_rows > 1u && regime_split == 0u )
			rows = rig->positions_total - position < maximum_rows ? rig->positions_total - position : maximum_rows;
		if ( rows == 0u )
			status = 101;
		else
		{
			RowsWave(rig,position,rows,tokens + position);
			status = RowsWalk(rig);
			(*waves_out)++;
		}
	}
	if ( status == 0 && (cudaMemcpy(boundary_out,rig->boundary,(uint64_t)rig->positions_total * ROWS_BOUNDARY * sizeof(uint16_t),cudaMemcpyDeviceToHost) != cudaSuccess ||
		cudaMemcpy(kv_out,rig->kv_cache,rig->kv_bytes,cudaMemcpyDeviceToHost) != cudaSuccess ||
		cudaMemcpy(index_out,rig->index_cache,rig->index_bytes,cudaMemcpyDeviceToHost) != cudaSuccess) )
		status = 102;
	free(lane_ids);
	free(positions);
	free(tokens);
	if ( status != 0 )
		fprintf(stderr,"glm52_prefill_rows_parity rows=%u split=%u position=%u status=%d cuda=%s\n",maximum_rows,regime_split,position,status,cudaGetErrorString(cudaGetLastError()));
	return(status);
}

static float RowsBf16(uint16_t value)
{
	uint32_t bits = (uint32_t)value << 16u;
	float result;
	memcpy(&result,&bits,sizeof(result));
	return(result);
}

static uint32_t RowsCompareBoundary(const RowsRig *rig,const uint16_t *a,const uint16_t *b,uint32_t first,uint32_t count,uint32_t *first_differing,float *max_difference)
{
	uint32_t position,differing = 0u;
	uint64_t index;
	float difference;
	*first_differing = UINT32_MAX;
	*max_difference = 0.0f;
	for (position=first; position<first + count && position<rig->positions_total; position++)
	{
		const uint16_t *row_a = a + (uint64_t)position * ROWS_BOUNDARY;
		const uint16_t *row_b = b + (uint64_t)position * ROWS_BOUNDARY;
		if ( memcmp(row_a,row_b,ROWS_BOUNDARY * sizeof(uint16_t)) == 0 )
			continue;
		differing++;
		if ( *first_differing == UINT32_MAX )
			*first_differing = position;
		for (index=0u; index<ROWS_BOUNDARY; index++)
		{
			difference = fabsf(RowsBf16(row_a[index]) - RowsBf16(row_b[index]));
			if ( difference > *max_difference )
				*max_difference = difference;
		}
	}
	return(differing);
}

static uint32_t RowsNonzero(const RowsRig *rig)
{
	uint64_t index,count = (uint64_t)rig->positions_total * ROWS_BOUNDARY;
	for (index=0u; index<count; index++)
		if ( (rig->reference_boundary[index] & 0x7fffu) != 0u )
			return(1u);
	return(0u);
}

static uint32_t RowsIndexWritten(const RowsRig *rig,const uint8_t *index_cache,uint32_t position)
{
	uint64_t offset = (uint64_t)(position / SPARK_GLM52_VKV_PAGE_SLOTS) * SPARK_GLM52_VINDEX_PAGE_BYTES + (uint64_t)(position % SPARK_GLM52_VKV_PAGE_SLOTS) * SPARK_GLM52_VINDEX_SLOT_BYTES;
	uint32_t index;
	(void)rig;
	for (index=0u; index<SPARK_GLM52_VINDEX_SLOT_BYTES; index++)
		if ( index_cache[offset + index] != 0u )
			return(1u);
	return(0u);
}

static int RowsVerify(RowsRig *rig,uint32_t anchor,uint32_t depth,uint32_t adversary)
{
	uint32_t tokens[ROWS_MAX],row,first,differing,position;
	float difference;
	SparkGlm52ExecutionSlot *slot = &rig->fixture.slot;
	SparkRowLayoutDenseLaneContext lanes;
	RowsRegimeContext regime;
	uint32_t lane_ids[ROWS_MAX],positions[ROWS_MAX];
	int status;
	for (row=0u; row<depth; row++)
	{
		lane_ids[row] = 0u;
		positions[row] = anchor + row;
	}
	lanes.lane_count = 1u;
	regime.positions = positions;
	depth = SparkRowLayoutRoundSpanWaveRowCount(0u,depth,lane_ids,SparkRowLayoutDenseLaneOrdinal,&lanes,RowsRegime,&regime,depth);
	if ( depth < 2u )
	{
		printf("glm52_prefill_rows_parity verify anchor=%u depth=%u skipped: the next position starts another attention regime\n",anchor,depth);
		return(0);
	}
	if ( cudaMemcpy(rig->kv_cache,rig->reference_kv,rig->kv_bytes,cudaMemcpyHostToDevice) != cudaSuccess ||
		cudaMemcpy(rig->index_cache,rig->reference_index,rig->index_bytes,cudaMemcpyHostToDevice) != cudaSuccess ||
		cudaMemcpy(rig->boundary,rig->reference_boundary,(uint64_t)rig->positions_total * ROWS_BOUNDARY * sizeof(uint16_t),cudaMemcpyHostToDevice) != cudaSuccess ||
		cudaMemset(slot->context_lengths,0,ROWS_MAX * sizeof(uint32_t)) != cudaSuccess )
		return(200);
	for (row=0u; row<depth; row++)
		tokens[row] = adversary != 0u && row != 0u ? (RowsToken(anchor + row) + 1u + row) % SPARK_GLM52_VALIDATION_EMBED_ROWS : RowsToken(anchor + row);
	RowsWave(rig,anchor,depth,tokens);
	status = RowsWalk(rig);
	if ( status != 0 )
		return(201);
	if ( cudaMemcpy(rig->mode_boundary,rig->boundary,(uint64_t)rig->positions_total * ROWS_BOUNDARY * sizeof(uint16_t),cudaMemcpyDeviceToHost) != cudaSuccess )
		return(202);
	if ( adversary == 0u )
	{
		differing = RowsCompareBoundary(rig,rig->mode_boundary,rig->reference_boundary,anchor,depth,&first,&difference);
		printf("glm52_prefill_rows_parity verify=oracle anchor=%u depth=%u differing_rows=%u first=%d max_abs=%.6g %s\n",anchor,depth,differing,first == UINT32_MAX ? -1 : (int)first,difference,differing == 0u ? "BIT-EXACT" : "DIFFER");
		return(differing == 0u ? 0 : 1);
	}
	differing = RowsCompareBoundary(rig,rig->mode_boundary,rig->reference_boundary,anchor,1u,&first,&difference);
	row = RowsCompareBoundary(rig,rig->mode_boundary,rig->reference_boundary,anchor + 1u,depth - 1u,&first,&difference);
	printf("glm52_prefill_rows_parity verify=adversary anchor=%u depth=%u anchor_row=%s drafted_rows_differing=%u/%u %s\n",anchor,depth,differing == 0u ? "EQUAL" : "DIFFER",row,depth - 1u,differing == 0u && row == depth - 1u ? "SEPARATED" : "FAIL");
	if ( differing != 0u || row != depth - 1u )
		return(1);
	for (position=anchor + 1u; position<anchor + depth + ROWS_ADVERSARY_TAIL && position<rig->positions_total; position++)
	{
		tokens[0] = RowsToken(position);
		RowsWave(rig,position,1u,tokens);
		if ( RowsWalk(rig) != 0 )
			return(203);
	}
	if ( cudaMemcpy(rig->mode_boundary,rig->boundary,(uint64_t)rig->positions_total * ROWS_BOUNDARY * sizeof(uint16_t),cudaMemcpyDeviceToHost) != cudaSuccess )
		return(204);
	differing = RowsCompareBoundary(rig,rig->mode_boundary,rig->reference_boundary,anchor + 1u,depth - 1u + ROWS_ADVERSARY_TAIL,&first,&difference);
	printf("glm52_prefill_rows_parity verify=rollback anchor=%u replayed=%u differing_rows=%u first=%d max_abs=%.6g %s\n",anchor,depth - 1u + ROWS_ADVERSARY_TAIL,differing,first == UINT32_MAX ? -1 : (int)first,difference,differing == 0u ? "BIT-EXACT" : "DIFFER");
	return(differing == 0u ? 0 : 1);
}

int main(int argc,char **argv)
{
	static const uint32_t widths[] = {2u,4u,8u,16u};
	static const uint32_t anchors[] = {5u,60u,100u,1500u,2044u,2047u,2060u,2090u};
	RowsRig *rig;
	uint32_t index,split,waves,differing,first,positions_total,missing,position;
	float difference;
	int failures = 0,exact_required;
	positions_total = argc > 1 ? (uint32_t)strtoul(argv[1],0,10) : 2112u;
	exact_required = argc > 2 ? atoi(argv[2]) : 1;
	rig = (RowsRig *)calloc(1u,sizeof(*rig));
	if ( rig == 0 || positions_total < 80u || positions_total > 4096u )
		return(1);
	if ( SparkGlm52ValFixtureSetup(&rig->fixture) != 0 || RowsSetup(rig,positions_total) != 0 )
	{
		fprintf(stderr,"glm52_prefill_rows_parity setup failed\n");
		return(1);
	}
	if ( RowsRun(rig,1u,0u,rig->reference_boundary,rig->reference_kv,rig->reference_index,&waves) != 0 )
		return(1);
	if ( RowsNonzero(rig) == 0u )
	{
		fprintf(stderr,"glm52_prefill_rows_parity FAIL reference outputs are all zero\n");
		return(1);
	}
	missing = 0u;
	for (position=0u; position<positions_total; position++)
		missing += RowsIndexWritten(rig,rig->reference_index,position) == 0u ? 1u : 0u;
	printf("glm52_prefill_rows_parity reference positions=%u waves=%u layers=%u..%u split_threshold=%u index_keys_missing=%u %s\n",
		positions_total,waves,ROWS_FIRST_LAYER,ROWS_FIRST_LAYER + ROWS_LAYERS - 1u,ROWS_SPLIT_THRESHOLD,missing,missing == 0u ? "INDEX-STORED" : "FAIL");
	failures += missing != 0u ? 1 : 0;
	for (split=0u; split<2u; split++)
		for (index=0u; index<sizeof(widths)/sizeof(widths[0]); index++)
		{
			if ( RowsRun(rig,widths[index],split,rig->mode_boundary,rig->mode_kv,rig->mode_index,&waves) != 0 )
				return(1);
			differing = RowsCompareBoundary(rig,rig->mode_boundary,rig->reference_boundary,0u,positions_total,&first,&difference);
			printf("glm52_prefill_rows_parity prefill rows<=%u regime_split=%u waves=%u differing_rows=%u first=%d max_abs=%.6g kv=%s index=%s %s\n",
				widths[index],split,waves,differing,first == UINT32_MAX ? -1 : (int)first,difference,
				memcmp(rig->mode_kv,rig->reference_kv,rig->kv_bytes) == 0 ? "EQUAL" : "DIFFER",
				memcmp(rig->mode_index,rig->reference_index,rig->index_bytes) == 0 ? "EQUAL" : "DIFFER",
				differing == 0u ? "BIT-EXACT" : (widths[index] > LM_SKINNY_ROWS || split == 0u ? "DIFFER-EXPECTED" : "DIFFER"));
			failures += split != 0u && exact_required != 0 && widths[index] <= LM_SKINNY_ROWS && differing != 0u ? 1 : 0;
			failures += split == 0u && widths[index] > 2u && differing == 0u ? 1 : 0;
		}
	for (index=0u; index<sizeof(anchors)/sizeof(anchors[0]); index++)
	{
		if ( anchors[index] + ROWS_VERIFY_DEPTH + ROWS_ADVERSARY_TAIL > positions_total )
			continue;
		failures += RowsVerify(rig,anchors[index],ROWS_VERIFY_DEPTH,0u) != 0 && exact_required != 0 ? 1 : 0;
		failures += RowsVerify(rig,anchors[index],ROWS_VERIFY_DEPTH,1u) != 0 ? 1 : 0;
	}
	printf("glm52_prefill_rows_parity %s\n",failures == 0 ? "PASS" : "FAIL");
	return(failures == 0 ? 0 : 1);
}
