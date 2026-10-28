#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "inference/runner/device_stage_model.h"
#include "inference/runner/device_stage_runner.h"
#include "sparkpipe/spark_weight_codec.h"
#include "sparkpipe/spark_weightd_manifest.h"

#define STAGE_SLICE_ALIGNMENT 256u

static uint64_t StagePageBytes(const SparkDeviceStageShape *shape)
{
	return((uint64_t)SPARK_DEVICE_RUNNER_PAGE_SLOTS * 2u * shape->kv_heads * shape->kv_head_dimension * 2u);
}

SparkStatus SparkDeviceStageRefuse(const SparkDeviceStage *stage, SparkStatus status, const char *what)
{
	fprintf(stderr,"%s: %s\n",stage != 0 && stage->shape != 0 ? stage->shape->tag : "device_stage",what);
	return(status);
}

SparkStatus SparkDeviceStageOpen(SparkDeviceStage *stage, const SparkDeviceStageShape *shape, const SparkStageRunnerModelOpen *request)
{
	const SparkStageRunnerConfiguration *configuration = request->configuration;
	uint32_t degree = 0u, rank = 0u;
	SparkStatus status;
	stage->shape = shape;
	if ( configuration->stage_count != 1u || configuration->tp_degree != 1u || configuration->max_input_row_count == 0u ||
		configuration->state_budget_bytes == 0u || configuration->kv_pages_per_sequence == 0u || configuration->resident_sequence_capacity == 0u )
		return(SparkDeviceStageRefuse(stage,SPARK_STATUS_INVALID_ARGUMENT,"the device model runs one TP1 stage with rows, sequences, KV pages and a state budget"));
	if ( configuration->linear_weight_codec != SPARK_WEIGHT_CODEC_BF16 )
		return(SparkDeviceStageRefuse(stage,SPARK_STATUS_UNSUPPORTED,"the device model reads the publisher's BF16 projections; this codec has no Metal kernels"));
	stage->services = request->services;
	stage->max_rows = configuration->max_input_row_count;
	stage->verify_rows = stage->max_rows < SPARK_DEVICE_STAGE_VERIFY_ROWS ? stage->max_rows : SPARK_DEVICE_STAGE_VERIFY_ROWS;
	stage->slots = configuration->resident_sequence_capacity;
	stage->pages_per_sequence = configuration->kv_pages_per_sequence;
	stage->max_positions = configuration->kv_pages_per_sequence * SPARK_DEVICE_RUNNER_PAGE_SLOTS;
	status = SparkNamedPackOpen(configuration->rank_pack_path,SPARK_NAMED_PACK_MAGIC,SPARK_NAMED_PACK_VERSION,shape->pack_alignment,&stage->pack);
	if ( status != SPARK_STATUS_OK )
		return(SparkDeviceStageRefuse(stage,status,"cannot open the rank pack"));
	stage->pack_open = 1u;
	stage->pack_bytes = stage->pack.file_bytes;
	if ( SparkNamedPackConfigU32(&stage->pack,"tp_degree",&degree) != SPARK_STATUS_OK ||
		SparkNamedPackConfigU32(&stage->pack,"tp_rank",&rank) != SPARK_STATUS_OK )
		return(SparkDeviceStageRefuse(stage,SPARK_STATUS_PARSE_ERROR,"the pack configuration lacks its topology"));
	if ( degree != 1u || rank != 0u || degree != configuration->tp_degree || rank != configuration->tp_rank )
	{
		fprintf(stderr,"%s: pack is tp %u rank %u; the device model is one whole TP1 stage\n",shape->tag,degree,rank);
		return(SPARK_STATUS_TARGET_MISMATCH);
	}
	status = SparkDeviceOpsAcquire(&stage->ops);
	if ( status != SPARK_STATUS_OK )
		return(SparkDeviceStageRefuse(stage,status,"the device kernel library is not available"));
	return(SPARK_STATUS_OK);
}

SparkStatus SparkDeviceStageExpectConfig(SparkDeviceStage *stage, const SparkDeviceStageConfigValue *values, uint32_t count)
{
	uint32_t index, value;
	for ( index = 0u; index < count; index++ )
	{
		value = 0u;
		if ( SparkNamedPackConfigU32(&stage->pack,values[index].key,&value) != SPARK_STATUS_OK )
		{
			fprintf(stderr,"%s: the pack configuration lacks %s\n",stage->shape->tag,values[index].key);
			return(SPARK_STATUS_PARSE_ERROR);
		}
		if ( value != values[index].value )
		{
			fprintf(stderr,"%s: the pack's %s is %u; the device model needs %u\n",stage->shape->tag,values[index].key,value,values[index].value);
			return(SPARK_STATUS_TARGET_MISMATCH);
		}
	}
	return(SPARK_STATUS_OK);
}

SparkStatus SparkDeviceStageReserve(SparkDeviceStage *stage, uint64_t scratch_bytes, uint32_t recurrent_layers, const uint64_t *recurrent_widths,
	uint32_t recurrent_pool_count, uint64_t state_budget_bytes)
{
	uint64_t state_bytes;
	uint32_t pool;
	if ( recurrent_pool_count > SPARK_DEVICE_STAGE_MAX_RECURRENT_POOLS || (recurrent_pool_count != 0u && recurrent_layers == 0u) )
		return(SparkDeviceStageRefuse(stage,SPARK_STATUS_INVALID_ARGUMENT,"the recurrent state needs layers and at most four pools"));
	stage->scratch_bytes = scratch_bytes;
	stage->recurrent_layers = recurrent_layers;
	stage->recurrent_pool_count = recurrent_pool_count;
	for ( pool = 0u; pool < recurrent_pool_count; pool++ )
		stage->recurrent_pools[pool].width = recurrent_widths[pool];
	state_bytes = (uint64_t)stage->slots * SparkDeviceStageRecurrentBytes(stage);
	if ( scratch_bytes + state_bytes > state_budget_bytes )
	{
		fprintf(stderr,"%s: state %llu bytes is over the %llu-byte budget (rows %u, slots %u)\n",stage->shape->tag,
			(unsigned long long)(scratch_bytes + state_bytes),(unsigned long long)state_budget_bytes,stage->max_rows,stage->slots);
		return(SPARK_STATUS_CAPACITY_EXCEEDED);
	}
	return(SPARK_STATUS_OK);
}

SparkStatus SparkDeviceStageEntry(SparkDeviceStage *stage, const char *name, const char *kind, uint64_t *offset, uint64_t *bytes, uint32_t *rows)
{
	SparkNamedPackEntry entry;
	if ( SparkNamedPackEntryLoad(&stage->pack,name,&entry) != SPARK_STATUS_OK || strcmp(entry.kind,kind) != 0 || entry.shape_count < 1u )
	{
		fprintf(stderr,"%s: pack tensor %s missing or not %s\n",stage->shape->tag,name,kind);
		return(SPARK_STATUS_PARSE_ERROR);
	}
	*offset = stage->pack.payload_base + entry.payload_offset;
	*bytes = entry.bytes;
	if ( rows != 0 )
		*rows = entry.shape[0];
	return(SPARK_STATUS_OK);
}

SparkStatus SparkDeviceStageGeometry(SparkDeviceStage *stage, SparkStageRunnerModelGeometry *geometry)
{
	const SparkDeviceStageShape *shape = stage->shape;
	const uint64_t table_bytes = (uint64_t)shape->vocab * shape->hidden * 2u;
	uint32_t embed_rows = 0u, head_rows = 0u;
	SparkStatus status;
	memset(geometry,0,sizeof(*geometry));
	geometry->hidden = shape->hidden;
	geometry->vocab = shape->vocab;
	geometry->total_layers = shape->layers;
	geometry->first_layer = 0u;
	geometry->layer_count = shape->layers;
	geometry->rms_epsilon = shape->epsilon;
	geometry->pack_bytes = stage->pack.file_bytes;
	geometry->kv_layer_count = shape->kv_layers;
	geometry->kv_layer_page_bytes = StagePageBytes(shape);
	geometry->kv_shard.degree = 1u;
	geometry->kv_shard.rank = 0u;
	geometry->kv_shard.grain = 1u;
	geometry->recurrent_bytes = SparkDeviceStageRecurrentBytes(stage);
	geometry->head_norm_f32 = 1u;
	geometry->head_prenormed = 0u;
	status = SparkDeviceStageEntry(stage,"embed","bf16",&geometry->embed_offset,&geometry->embed_bytes,&embed_rows);
	if ( status == SPARK_STATUS_OK )
		status = SparkDeviceStageEntry(stage,"head_norm","f32",&geometry->head_norm_offset,&geometry->head_norm_bytes,0);
	if ( status == SPARK_STATUS_OK && shape->tied_head != 0u )
	{
		geometry->head_offset = geometry->embed_offset;
		geometry->head_bytes = geometry->embed_bytes;
		head_rows = embed_rows;
	}
	else if ( status == SPARK_STATUS_OK )
		status = SparkDeviceStageEntry(stage,"head","bf16",&geometry->head_offset,&geometry->head_bytes,&head_rows);
	if ( status != SPARK_STATUS_OK )
		return(status);
	geometry->logit_softcap = shape->logit_softcap;
	geometry->logit_scale = shape->logit_scale;
	geometry->embed_alias_first = shape->embed_alias_first;
	geometry->embed_alias_count = shape->embed_alias_count;
	geometry->embed_alias_row = shape->embed_alias_row;
	if ( shape->embed_alias_count != 0u && (shape->embed_alias_first >= shape->vocab || shape->embed_alias_count > shape->vocab - shape->embed_alias_first ||
		shape->embed_alias_row >= shape->vocab || shape->embed_alias_row - shape->embed_alias_first < shape->embed_alias_count) )
		return(SparkDeviceStageRefuse(stage,SPARK_STATUS_SCHEMA_ERROR,"the embedding alias must name tokens and a row of the vocabulary, the row outside the aliased tokens"));
	if ( embed_rows != shape->vocab || head_rows != shape->vocab || geometry->embed_bytes != table_bytes || geometry->head_bytes != table_bytes ||
		geometry->head_norm_bytes != (uint64_t)shape->hidden * 4u )
		return(SparkDeviceStageRefuse(stage,SPARK_STATUS_SCHEMA_ERROR,"the pack's embedding, head or head norm has the wrong shape"));
	geometry->embed_rows = embed_rows;
	geometry->head_rows = head_rows;
	return(SPARK_STATUS_OK);
}

SparkStatus SparkDeviceStageTensor(SparkDeviceStage *stage, const SparkStageRunnerDeviceWeights *weights, const char *name, const char *kind,
	uint64_t expected_bytes, SparkTensor *tensor)
{
	SparkDeviceRange range;
	uint64_t offset = 0u, bytes = 0u;
	SparkStatus status = SparkDeviceStageEntry(stage,name,kind,&offset,&bytes,0);
	if ( status != SPARK_STATUS_OK )
		return(status);
	if ( bytes != expected_bytes || offset + bytes > weights->pack_bytes || offset % 16u != 0u )
	{
		fprintf(stderr,"%s: pack tensor %s holds %llu bytes at %llu; the model needs %llu aligned bytes inside the pack\n",stage->shape->tag,name,
			(unsigned long long)bytes,(unsigned long long)offset,(unsigned long long)expected_bytes);
		return(SPARK_STATUS_SCHEMA_ERROR);
	}
	status = SparkDeviceResolve(weights->pack_address + offset,bytes,&range);
	if ( status != SPARK_STATUS_OK )
		return(SparkDeviceStageRefuse(stage,status,"the pack mapping does not hold a tensor's bytes"));
	tensor->range = range;
	tensor->bytes = bytes;
	return(SPARK_STATUS_OK);
}

SparkStatus SparkDeviceStageLayerWeight(SparkDeviceStage *stage, const SparkStageRunnerDeviceWeights *weights, uint32_t layer, const char *name,
	uint64_t rows, uint64_t columns, SparkDeviceWeight *weight)
{
	char key[SPARK_NAMED_PACK_MAX_NAME_BYTES], scale_key[SPARK_NAMED_PACK_MAX_NAME_BYTES + 8u];
	SparkNamedPackEntry entry;
	SparkStatus status;
	memset(weight,0,sizeof(*weight));
	snprintf(key,sizeof(key),"layers.%u.%s",layer,name);
	if ( SparkNamedPackEntryLoad(&stage->pack,key,&entry) != SPARK_STATUS_OK || strcmp(entry.kind,"fp8_e4m3") != 0 )
		return(SparkDeviceStageTensor(stage,weights,key,"bf16",rows * columns * 2u,&weight->values));
	snprintf(scale_key,sizeof(scale_key),"%s.scale",key);
	status = SparkDeviceStageTensor(stage,weights,key,"fp8_e4m3",rows * columns,&weight->values);
	if ( status == SPARK_STATUS_OK )
		status = SparkDeviceStageTensor(stage,weights,scale_key,"f32",rows * ((columns + SPARK_FP8_SCALE_K - 1u) / SPARK_FP8_SCALE_K) * sizeof(float),
			&weight->scale);
	return(status);
}

SparkStatus SparkDeviceStageLayerTensor(SparkDeviceStage *stage, const SparkStageRunnerDeviceWeights *weights, uint32_t layer, const char *name,
	const char *kind, uint64_t expected_bytes, SparkTensor *tensor)
{
	char key[SPARK_NAMED_PACK_MAX_NAME_BYTES];
	snprintf(key,sizeof(key),"layers.%u.%s",layer,name);
	return(SparkDeviceStageTensor(stage,weights,key,kind,expected_bytes,tensor));
}

uint64_t SparkDeviceStagePlan(const SparkDeviceBuffer *buffer, const SparkDeviceStageSlice *slices, uint32_t count)
{
	uint64_t cursor = 0u;
	uint32_t index;
	for ( index = 0u; index < count; index++ )
	{
		cursor = (cursor + STAGE_SLICE_ALIGNMENT - 1u) / STAGE_SLICE_ALIGNMENT * STAGE_SLICE_ALIGNMENT;
		if ( buffer != 0 )
			*slices[index].target = SparkTensorAt(buffer,cursor,slices[index].bytes);
		cursor += slices[index].bytes;
	}
	return(cursor);
}

SparkStatus SparkDeviceStageAllocate(SparkDeviceStage *stage, const SparkDeviceStageSlice *slices, uint32_t count)
{
	char label[96];
	uint32_t pool;
	SparkStatus status;
	if ( SparkDeviceStagePlan(0,slices,count) != stage->scratch_bytes )
		return(SparkDeviceStageRefuse(stage,SPARK_STATUS_INTERNAL_ERROR,"the scratch plan changed after the budget was reserved"));
	snprintf(label,sizeof(label),"%s scratch",stage->shape->tag);
	status = SparkDeviceAllocate(&stage->scratch_buffer,SPARK_DEVICE_SPACE_DEVICE_PRIVATE,stage->scratch_bytes,label);
	for ( pool = 0u; pool < stage->recurrent_pool_count && status == SPARK_STATUS_OK; pool++ )
	{
		snprintf(label,sizeof(label),"%s recurrent pool %u",stage->shape->tag,pool);
		status = SparkDeviceAllocate(&stage->recurrent_pools[pool].buffer,SPARK_DEVICE_SPACE_DEVICE_PRIVATE,
			(uint64_t)stage->recurrent_layers * stage->slots * stage->recurrent_pools[pool].width,label);
	}
	if ( status == SPARK_STATUS_OK )
	{
		snprintf(label,sizeof(label),"%s keep tables",stage->shape->tag);
		status = SparkDeviceAllocate(&stage->keep_tables,SPARK_DEVICE_SPACE_SHARED_COHERENT,
			((uint64_t)2u * stage->verify_rows + stage->slots + 1u) * sizeof(uint32_t),label);
	}
	if ( status != SPARK_STATUS_OK )
		return(SparkDeviceStageRefuse(stage,status,"cannot allocate the device scratch and recurrent state"));
	for ( pool = 0u; pool < stage->verify_rows; pool++ )
		((uint32_t *)stage->keep_tables.host)[pool] = pool;
	SparkDeviceStagePlan(&stage->scratch_buffer,slices,count);
	return(SPARK_STATUS_OK);
}

SparkTensor SparkDeviceStageRecurrent(const SparkDeviceStage *stage, uint32_t pool, uint32_t layer)
{
	const uint64_t bytes = (uint64_t)stage->slots * stage->recurrent_pools[pool].width;
	return(SparkTensorAt(&stage->recurrent_pools[pool].buffer,(uint64_t)layer * bytes,bytes));
}

uint64_t SparkDeviceStageRecurrentBytes(const SparkDeviceStage *stage)
{
	uint64_t width = 0u;
	uint32_t pool;
	for ( pool = 0u; pool < stage->recurrent_pool_count; pool++ )
		width += stage->recurrent_pools[pool].width;
	return((uint64_t)stage->recurrent_layers * width);
}

static SparkStatus StageResolve(const SparkDeviceStage *stage, const void *address, uint64_t bytes, SparkTensor *tensor)
{
	SparkDeviceRange range;
	SparkStatus status = SparkDeviceResolve((SparkDeviceAddress)(uintptr_t)address,bytes,&range);
	if ( status != SPARK_STATUS_OK )
		return(SparkDeviceStageRefuse(stage,status,"a step table is not device memory of the expected size"));
	tensor->range = range;
	tensor->bytes = bytes;
	return(SPARK_STATUS_OK);
}

SparkStatus SparkDeviceStageNoteKvRead(SparkDeviceStage *stage, uint32_t window, uint64_t position_bytes)
{
	uint32_t index;
	for ( index = 0u; index < stage->kv_read_count && stage->kv_reads[index].window != window; index++ )
		continue;
	if ( index == SPARK_DEVICE_STAGE_KV_WINDOWS )
		return(SparkDeviceStageRefuse(stage,SPARK_STATUS_UNSUPPORTED,"a stage's attention layers use at most four distinct windows"));
	if ( index == stage->kv_read_count )
	{
		stage->kv_reads[index].window = window;
		stage->kv_reads[index].position_bytes = 0u;
		stage->kv_read_count++;
	}
	stage->kv_reads[index].position_bytes += position_bytes;
	return(SPARK_STATUS_OK);
}

uint64_t SparkDeviceStageStateBytes(const SparkDeviceStage *stage, uint32_t context)
{
	uint64_t bytes = 2u * SparkDeviceStageRecurrentBytes(stage);
	uint32_t index;
	for ( index = 0u; index < stage->kv_read_count; index++ )
	{
		const uint32_t window = stage->kv_reads[index].window;
		bytes += (uint64_t)(window != 0u && window < context ? window : context) * stage->kv_reads[index].position_bytes;
	}
	return(bytes);
}

SparkStatus SparkDeviceStageStepTables(SparkDeviceStage *stage, const SparkStageRunnerStep *step, SparkDeviceStepTensors *tensors)
{
	SparkStatus status;
	uint32_t ordinal, groups;
	if ( step->rows == 0u || step->rows > stage->max_rows || step->sequences == 0u || step->sequences > stage->slots || stage->kv_attached == 0u ||
		step->host_positions == 0 || step->host_sequence_of_row == 0 || step->recurrent_index == 0 || step->sequence_row_begin == 0 )
		return(SparkDeviceStageRefuse(stage,SPARK_STATUS_INVALID_ARGUMENT,"a step needs rows and sequences within capacity, an attached KV pool and its row tables"));
	for ( ordinal = 0u, groups = 0u; ordinal < step->rows; ordinal++ )
	{
		if ( step->host_sequence_of_row[ordinal] >= stage->slots )
			return(SparkDeviceStageRefuse(stage,SPARK_STATUS_INVALID_ARGUMENT,"a row names a sequence outside the model's slots"));
		if ( ordinal == 0u || step->host_sequence_of_row[ordinal] != step->host_sequence_of_row[ordinal - 1u] )
			groups++;
	}
	if ( groups != step->sequences )
		return(SparkDeviceStageRefuse(stage,SPARK_STATUS_INVALID_ARGUMENT,"each step sequence must own one contiguous run of rows"));
	if ( step->commit == 0u && step->rows > stage->verify_rows )
		return(SparkDeviceStageRefuse(stage,SPARK_STATUS_INVALID_ARGUMENT,"a verify step holds more rows than the stage keeps for its replay"));
	stage->step_exact = step->commit == 0u ? 1u : 0u;
	tensors->row_identity = SparkTensorAt(&stage->keep_tables,0u,(uint64_t)stage->verify_rows * sizeof(uint32_t));
	tensors->rows = step->rows;
	tensors->host_positions = step->host_positions;
	tensors->host_sequence_of_row = step->host_sequence_of_row;
	status = StageResolve(stage,step->hidden_bf16,(uint64_t)step->rows * stage->shape->hidden * 2u,&tensors->hidden);
	if ( status == SPARK_STATUS_OK )
		status = StageResolve(stage,step->positions,(uint64_t)step->rows * 4u,&tensors->positions);
	if ( status == SPARK_STATUS_OK )
		status = StageResolve(stage,step->sequence_of_row,(uint64_t)step->rows * 4u,&tensors->sequence_of_row);
	if ( status == SPARK_STATUS_OK )
		status = StageResolve(stage,step->recurrent_index,(uint64_t)step->sequences * 4u,&tensors->sequences.state_index);
	if ( status == SPARK_STATUS_OK )
		status = StageResolve(stage,step->sequence_row_begin,((uint64_t)step->sequences + 1u) * 4u,&tensors->sequences.row_begin);
	tensors->sequences.sequences = step->sequences;
	tensors->sequences.commit = step->commit;
	tensors->route_override_width = 0u;
	if ( status == SPARK_STATUS_OK && step->route_override != 0 )
	{
		if ( step->route_override_width == 0u || step->route_override_width > SPARK_ROUTE_MAX_TOP_K )
			return(SparkDeviceStageRefuse(stage,SPARK_STATUS_INVALID_ARGUMENT,"a route override needs between 1 and 16 experts per row"));
		status = StageResolve(stage,step->route_override,(uint64_t)stage->shape->layers * step->rows * step->route_override_width * sizeof(uint32_t),
			&tensors->route_override);
		tensors->route_override_width = step->route_override_width;
	}
	return(status);
}

SparkStatus SparkDeviceStageKeepTables(SparkDeviceStage *stage, const SparkStageRunnerStep *step, const uint32_t *kept, SparkDeviceKeepTensors *keep)
{
	uint32_t *table = (uint32_t *)stage->keep_tables.host, *ids = table + stage->verify_rows, *begin = ids + stage->verify_rows;
	const uint32_t *host_kept = 0, *host_begin = 0;
	SparkDeviceRange range;
	uint32_t sequence, row, count = 0u;
	SparkStatus status;
	if ( step->commit != 0u || step->rows > stage->verify_rows || step->sequences > stage->slots )
		return(SparkDeviceStageRefuse(stage,SPARK_STATUS_INVALID_ARGUMENT,"keep_rows follows a verify step within the stage's verify rows"));
	status = SparkDeviceResolve((SparkDeviceAddress)(uintptr_t)kept,(uint64_t)step->sequences * sizeof(uint32_t),&range);
	if ( status == SPARK_STATUS_OK )
		status = SparkDeviceHostOf(range,(uint64_t)step->sequences * sizeof(uint32_t),(void **)&host_kept);
	if ( status == SPARK_STATUS_OK )
		status = SparkDeviceResolve((SparkDeviceAddress)(uintptr_t)step->sequence_row_begin,((uint64_t)step->sequences + 1u) * sizeof(uint32_t),&range);
	if ( status == SPARK_STATUS_OK )
		status = SparkDeviceHostOf(range,((uint64_t)step->sequences + 1u) * sizeof(uint32_t),(void **)&host_begin);
	if ( status != SPARK_STATUS_OK )
		return(SparkDeviceStageRefuse(stage,status,"keep_rows needs host-visible kept counts and row tables"));
	for ( sequence = 0u; sequence < step->sequences; sequence++ )
	{
		if ( host_kept[sequence] > host_begin[sequence + 1u] - host_begin[sequence] )
			return(SparkDeviceStageRefuse(stage,SPARK_STATUS_INVALID_ARGUMENT,"a sequence keeps more rows than its verify step held"));
		begin[sequence] = count;
		for ( row = 0u; row < host_kept[sequence]; row++ )
			ids[count++] = host_begin[sequence] + row;
	}
	begin[step->sequences] = count;
	keep->rows = count;
	keep->frame_rows = step->rows;
	keep->kept_rows = SparkTensorAt(&stage->keep_tables,(uint64_t)stage->verify_rows * sizeof(uint32_t),(uint64_t)stage->verify_rows * sizeof(uint32_t));
	keep->sequences.row_begin = SparkTensorAt(&stage->keep_tables,(uint64_t)2u * stage->verify_rows * sizeof(uint32_t),
		((uint64_t)step->sequences + 1u) * sizeof(uint32_t));
	keep->sequences.sequences = step->sequences;
	keep->sequences.commit = 1u;
	return(StageResolve(stage,step->recurrent_index,(uint64_t)step->sequences * sizeof(uint32_t),&keep->sequences.state_index));
}

SparkOpsPaged SparkDeviceStagePaged(const SparkDeviceStage *stage, const SparkDeviceStepTensors *tensors, uint32_t kv_layer)
{
	SparkOpsPaged paged;
	paged.pool = SparkTensorSlice(stage->kv_pool,(uint64_t)kv_layer * stage->kv_layer_stride,stage->kv_layer_stride);
	paged.page_table = stage->page_table;
	paged.sequence_of_row = tensors->sequence_of_row;
	paged.positions = tensors->positions;
	paged.page_slots = SPARK_DEVICE_RUNNER_PAGE_SLOTS;
	paged.pages_per_sequence = stage->pages_per_sequence;
	paged.pool_pages = stage->pool_pages;
	paged.kv_heads = stage->shape->kv_heads;
	return(paged);
}

void SparkDeviceStageLayerDone(const SparkDeviceStage *stage, uint32_t layer)
{
	if ( stage->services != 0 && stage->services->layer_done != 0 )
		stage->services->layer_done(stage->services->context,layer);
}

void SparkDeviceStageClose(void *model)
{
	SparkDeviceStage *stage = (SparkDeviceStage *)model;
	uint32_t pool;
	if ( stage == 0 )
		return;
	if ( stage->scratch_buffer.handle != 0u )
		(void)SparkDeviceFree(&stage->scratch_buffer);
	if ( stage->keep_tables.handle != 0u )
		(void)SparkDeviceFree(&stage->keep_tables);
	for ( pool = 0u; pool < stage->recurrent_pool_count; pool++ )
		if ( stage->recurrent_pools[pool].buffer.handle != 0u )
			(void)SparkDeviceFree(&stage->recurrent_pools[pool].buffer);
	if ( stage->ops != 0 )
		SparkDeviceOpsRelease(stage->ops);
	if ( stage->pack_open != 0u )
		SparkNamedPackClose(&stage->pack);
	free(model);
}

SparkStatus SparkDeviceStageManifestCheck(void *model, const struct SparkWeightdManifest *manifest)
{
	if ( manifest == 0 || manifest->range_count != 0u )
		return(SparkDeviceStageRefuse((const SparkDeviceStage *)model,SPARK_STATUS_SCHEMA_ERROR,"a dense pack's weightd manifest carries no expert ranges"));
	return(SPARK_STATUS_OK);
}

SparkStatus SparkDeviceStageBindWeightd(void *model, struct SparkWeightdLazyPack *lazy_pack, SparkStageRunnerModelGeometry *geometry)
{
	(void)lazy_pack;
	(void)geometry;
	return(SparkDeviceStageRefuse((const SparkDeviceStage *)model,SPARK_STATUS_UNSUPPORTED,
		"weightd lazy packs are CUDA arenas; the device model binds the device runner's mapping through bind_device"));
}

SparkStatus SparkDeviceStageSideband(void *model, uint32_t to_buffer, const void *hidden, void *buffer, uint64_t bytes, uint32_t rows,
	void *device_queue)
{
	(void)to_buffer; (void)hidden; (void)buffer; (void)bytes; (void)rows; (void)device_queue;
	return(SparkDeviceStageRefuse((const SparkDeviceStage *)model,SPARK_STATUS_UNSUPPORTED,"the device model has no sideband"));
}

SparkStatus SparkDeviceStageAttachKv(void *model, const SparkStageRunnerKv *kv)
{
	SparkDeviceStage *stage = (SparkDeviceStage *)model;
	const uint64_t page_bytes = StagePageBytes(stage->shape);
	SparkStatus status;
	if ( kv == 0 || kv->layer_count != stage->shape->kv_layers || kv->layer_page_bytes != page_bytes || kv->sequence_count > stage->slots ||
		kv->second_pool != 0 || kv->page_table_stride != stage->pages_per_sequence || kv->pool_page_count == 0u ||
		kv->layer_stride_bytes < (uint64_t)kv->pool_page_count * page_bytes || kv->context_shard.degree != 1u )
	{
		fprintf(stderr,"%s: the KV pool must hold %u layers of %u-slot pages for the model's slots, unsharded\n",stage->shape->tag,
			stage->shape->kv_layers,SPARK_DEVICE_RUNNER_PAGE_SLOTS);
		return(SPARK_STATUS_INVALID_ARGUMENT);
	}
	status = StageResolve(stage,kv->pool,(uint64_t)kv->layer_count * kv->layer_stride_bytes,&stage->kv_pool);
	if ( status == SPARK_STATUS_OK )
		status = StageResolve(stage,kv->page_table,(uint64_t)kv->sequence_count * kv->page_table_stride * 4u,&stage->page_table);
	if ( status != SPARK_STATUS_OK )
		return(status);
	stage->kv_layer_stride = kv->layer_stride_bytes;
	stage->pool_pages = kv->pool_page_count;
	stage->kv_attached = 1u;
	return(SPARK_STATUS_OK);
}

SparkStatus SparkDeviceStageRecurrentCopy(void *model, uint32_t to_buffer, uint32_t slot, void *buffer, uint64_t bytes, void *device_queue)
{
	SparkDeviceStage *stage = (SparkDeviceStage *)model;
	SparkDeviceQueue *queue = (SparkDeviceQueue *)device_queue;
	SparkTensor packed;
	uint64_t cursor = 0u;
	uint32_t pool, layer;
	SparkStatus status;
	if ( slot >= stage->slots || bytes != SparkDeviceStageRecurrentBytes(stage) || bytes == 0u || queue == 0 )
		return(SparkDeviceStageRefuse(stage,SPARK_STATUS_INVALID_ARGUMENT,"recurrent copy needs a slot, the model's recurrent size and a queue"));
	status = StageResolve(stage,buffer,bytes,&packed);
	for ( pool = 0u; pool < stage->recurrent_pool_count && status == SPARK_STATUS_OK; pool++ )
	{
		const uint64_t width = stage->recurrent_pools[pool].width;
		for ( layer = 0u; layer < stage->recurrent_layers && status == SPARK_STATUS_OK; layer++ )
		{
			SparkDeviceRange resident, staged;
			resident.buffer = &stage->recurrent_pools[pool].buffer;
			resident.offset = ((uint64_t)layer * stage->slots + slot) * width;
			staged.buffer = packed.range.buffer;
			staged.offset = packed.range.offset + cursor;
			status = to_buffer != 0u ? SparkDeviceCopy(queue,staged,resident,width) : SparkDeviceCopy(queue,resident,staged,width);
			cursor += width;
		}
	}
	return(status);
}

SparkStatus SparkDeviceStageResetSlot(void *model, uint32_t slot, void *device_queue)
{
	SparkDeviceStage *stage = (SparkDeviceStage *)model;
	SparkDeviceQueue *queue = (SparkDeviceQueue *)device_queue;
	uint32_t pool, layer;
	SparkStatus status = SPARK_STATUS_OK;
	if ( slot >= stage->slots || queue == 0 )
		return(SparkDeviceStageRefuse(stage,SPARK_STATUS_INVALID_ARGUMENT,"reset needs a slot and a queue"));
	for ( layer = 0u; layer < stage->recurrent_layers && status == SPARK_STATUS_OK; layer++ )
		for ( pool = 0u; pool < stage->recurrent_pool_count && status == SPARK_STATUS_OK; pool++ )
		{
			SparkDeviceRange resident;
			resident.buffer = &stage->recurrent_pools[pool].buffer;
			resident.offset = ((uint64_t)layer * stage->slots + slot) * stage->recurrent_pools[pool].width;
			status = SparkDeviceFill(queue,resident,0u,stage->recurrent_pools[pool].width);
		}
	return(status);
}

const void *SparkDeviceStageProbeBuffers(void *model)
{
	return(model);
}

void SparkDeviceStageReport(void *model, uint32_t rank)
{
	const SparkDeviceStage *stage = (const SparkDeviceStage *)model;
	fprintf(stderr,"DEVICE-STAGE tag=%s rank=%u scratch_bytes=%llu recurrent_bytes=%llu slots=%u rows=%u positions=%u\n",stage->shape->tag,rank,
		(unsigned long long)stage->scratch_bytes,(unsigned long long)((uint64_t)stage->slots * SparkDeviceStageRecurrentBytes(stage)),stage->slots,
		stage->max_rows,stage->max_positions);
}

SparkStatus SparkDeviceStageLane(void *model, uint32_t lane, const SparkStageRunnerStep *step, const SparkStageRunnerServices *services,
	void **lane_model)
{
	(void)lane; (void)step; (void)services; (void)lane_model;
	return(SparkDeviceStageRefuse((const SparkDeviceStage *)model,SPARK_STATUS_UNSUPPORTED,"the device runner has one execution lane"));
}
