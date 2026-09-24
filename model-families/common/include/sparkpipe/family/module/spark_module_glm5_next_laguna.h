#pragma once

static SparkStatus SPARK_FAMILY(PackValidateRanges)(
	const SPARK_FAMILY(StagePackEntry) *entries,
	uint32_t entry_count)
{
	SPARK_FAMILY(PackRange) left[2],right[2];
	uint32_t left_index,right_index,left_part,right_part;
	for (left_index=0u; left_index<entry_count; left_index++)
	{
		left[0].offset = entries[left_index].payload_offset;
		left[0].bytes = entries[left_index].payload_bytes;
		left[1].offset = entries[left_index].scale_offset;
		left[1].bytes = entries[left_index].scale_bytes;
		for (right_index=left_index + 1u; right_index<entry_count; right_index++)
		{
			right[0].offset = entries[right_index].payload_offset;
			right[0].bytes = entries[right_index].payload_bytes;
			right[1].offset = entries[right_index].scale_offset;
			right[1].bytes = entries[right_index].scale_bytes;
			for (left_part=0u; left_part<2u; left_part++)
				for (right_part=0u; right_part<2u; right_part++)
					if ( SPARK_FAMILY(PackRangesOverlap)(&left[left_part],&right[right_part]) != 0u )
						SPARK_FAIL(SPARK_STATUS_SCHEMA_ERROR);
		}
		if ( SPARK_FAMILY(PackRangesOverlap)(&left[0],&left[1]) != 0u )
			SPARK_FAIL(SPARK_STATUS_SCHEMA_ERROR);
	}
	return(SPARK_STATUS_OK);
}

static SparkStatus SPARK_FAMILY(ManifestPlane)(const SparkWeightdManifest *manifest,const SPARK_FAMILY(StagePackEntry) *entry,uint32_t plane)
{
	const SparkWeightdRangeGroup *group;
	const SparkWeightdRange *range;
	uint64_t offset,bytes,per;
	uint32_t expert,index,kind;
	offset = plane == 0u ? entry->payload_offset : entry->scale_offset;
	bytes = plane == 0u ? entry->payload_bytes : entry->scale_bytes;
	if ( entry->group_count == 0u || bytes == 0u || bytes % entry->group_count != 0u )
		SPARK_FAIL(SPARK_STATUS_SCHEMA_ERROR);
	per = (bytes / entry->group_count);
	kind = ((entry->tensor_kind * 2u) + plane);
	for (expert=0u; expert<entry->group_count; expert++)
	{
		group = SparkWeightdManifestFind(manifest,entry->layer_index,expert);
		if ( group == 0 )
			SPARK_FAIL(SPARK_STATUS_SCHEMA_ERROR);
		for (index=0u; index<group->range_count; index++)
		{
			range = &manifest->ranges[group->first_range + index];
			if ( range->kind == kind )
				break;
		}
		if ( index == group->range_count || range->offset != (offset + ((uint64_t)expert * per)) || range->bytes != per )
			SPARK_FAIL(SPARK_STATUS_SCHEMA_ERROR);
	}
	return(SPARK_STATUS_OK);
}

static SparkStatus SPARK_FAMILY(AllocateBytes)(
	SPARK_FAMILY(ModuleState) *state,
	uint64_t count,
	uint64_t width,
	uint64_t element_bytes,
	void **pointer)
{
	uint64_t bytes;
	if ( state == 0 || pointer == 0 || count == 0u || width == 0u || element_bytes == 0u || count > UINT64_MAX / width || count * width > UINT64_MAX / element_bytes )
		SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
	bytes = count * width * element_bytes;
	return(SparkStageModuleDeviceAllocate(&state->ledger,bytes,pointer));
}

static SparkStatus SPARK_FAMILY(DevicePageCopy)(
	void *context,
	uint32_t direction,
	uintptr_t device_address,
	void *host_address,
	uint64_t bytes)
{
	SPARK_FAMILY(ModuleState) *state;
	cudaError_t error;
	state = (SPARK_FAMILY(ModuleState) *)context;
	if ( state == 0 )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	if ( direction == SPARK_KV_PAGE_STORE_COPY_DEVICE_TO_HOST )
		error = cudaMemcpy(host_address,(const void *)device_address,(size_t)bytes,cudaMemcpyDeviceToHost);
	else if ( direction == SPARK_KV_PAGE_STORE_COPY_HOST_TO_DEVICE )
		error = cudaMemcpy((void *)device_address,host_address,(size_t)bytes,cudaMemcpyHostToDevice);
	else
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	return(SparkStageModuleCudaStatus(SPARK_FAMILY_CONST(MODULE_TAG),error,"kv_page_copy"));
}

static uint32_t SPARK_FAMILY(PrefixRestorePending)(const SparkKvLaneTransaction *owner)
{
	return(owner != 0 && (owner->mutation_flags & SPARK_KV_PAGE_CACHE_MUTATION_BOUND_SEQUENCE) != 0u && (owner->lane.flags & SPARK_MODEL_DRIVER_CACHE_LANE_FLAG_PREFIX) != 0u && owner->lane.sequence_position != 0u);
}

static void SPARK_FAMILY(T1Wave)(const SPARK_FAMILY(CudaWave) *wave)
{
	uint32_t i;
	if ( SPARK_FAMILY(T1Enabled)() == 0 || wave == 0 || wave->tp_rank != 0u ||
	    wave->host_resident_slots == 0 || wave->host_token_ids == 0 ||
	    wave->host_positions == 0 )
		return;
	fprintf(stderr,"G5N-T1 wave seq%u rows=%u",wave->host_resident_slots[0],wave->row_count);
	for ( i = 0u; i < wave->row_count; i++ )
		fprintf(stderr," pos%u=%u",wave->host_positions[i],wave->host_token_ids[i]);
	fputc('\n',stderr);
}

static SparkStatus SPARK_FAMILY(StageHostBatch)(
	const SPARK_FAMILY(ModuleState) *state,
	SPARK_FAMILY(ExecutionSlot) *slot,
	const SPARK_FAMILY(ResidentDecodeStageBatchView) *batch)
{
	uint32_t row;
	if ( state == 0 || slot == 0 || batch == 0 || batch->row_count > SPARK_FAMILY_CONST(RESIDENT_DECODE_STAGE_MAX_INPUT_ROW_COUNT) )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	for (row=0u; row<batch->row_count; row++)
	{
		if ( batch->row_positions[row] >= UINT32_MAX )
			SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
		slot->host_resident_slots[row] = batch->row_resident_slots[row];
		slot->host_positions[row] = (uint32_t)batch->row_positions[row];
		if ( state->owns_embedding != 0u )
			slot->host_token_ids[row] = batch->token_ids[row];
	}
	memset(slot->host_kv_access_error,0,SPARK_FAMILY_CONST(KV_ACCESS_ERROR_WORD_COUNT) * sizeof(uint32_t));
	return(SPARK_STATUS_OK);
}

static SparkStatus SPARK_FAMILY(UploadPageTables)(SPARK_FAMILY(ModuleState) *state,const SPARK_FAMILY(AsyncCompletion) *async,void *stream)
{
	uint32_t lane,resident;
	uint64_t offset,bytes;
	cudaError_t error;
	for (lane=0u; lane<async->lane_count; lane++)
	{
		resident = async->lane_indices[lane];
		offset = ((uint64_t)resident * state->pages_per_sequence);
		bytes = ((uint64_t)state->kv_lane_transactions[resident].page_count * sizeof(uint32_t));
		if ( memcmp(state->page_table_shadow + offset,state->kv_lane_physical_pages + offset,bytes) == 0 )
			continue;
		error = cudaMemcpyAsync(state->page_table + offset,state->kv_lane_physical_pages + offset,bytes,cudaMemcpyHostToDevice,(cudaStream_t)stream);
		if ( error != cudaSuccess )
			return(SparkStageModuleCudaStatus(SPARK_FAMILY_CONST(MODULE_TAG),error,"page_table_update"));
		memcpy(state->page_table_shadow + offset,state->kv_lane_physical_pages + offset,bytes);
	}
	return(SPARK_STATUS_OK);
}
